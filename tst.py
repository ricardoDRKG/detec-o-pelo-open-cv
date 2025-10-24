import cv2
import mediapipe as mp
import numpy as np
import math
import time
from datetime import datetime
import os

# Define o limiar de visibilidade para considerar um ponto válido
VISIBILITY_THRESHOLD = 0.6

# Inicializa as soluções do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


# --- FUNÇÕES AUXILIARES GLOBAIS ---

# Função 1: Checa Visibilidade e Retorna Coordenadas
def get_landmark_coords(landmarks, landmark_index):
    """
    Retorna as coordenadas [x, y, z] se o ponto for visível, caso contrário, retorna None.
    """
    landmark = landmarks[landmark_index]
    if landmark.visibility > VISIBILITY_THRESHOLD:
        return [landmark.x, landmark.y, landmark.z]
    return None  # Indica que o ponto não é confiável ou visível


# Função 2: Inferência de Ponto (Região Abaixo das Costelas)
def infer_costela_point(p_shoulder, p_hip):
    """Calcula um ponto médio ponderado entre o ombro e o quadril."""
    if p_shoulder is not None and p_hip is not None:
        # Pondera o ponto para ficar mais próximo da cintura (60% do HIP)
        return [
            (p_shoulder[0] + p_hip[0]) / 2,
            (p_shoulder[1] * 0.4 + p_hip[1] * 0.6) / (0.4 + 0.6),
            (p_shoulder[2] + p_hip[2]) / 2
        ]
    return None


# Função 3: Converte Coordenadas Normalizadas para Pixel (Para Visualização)
def normalize_to_pixel(point, w, h):
    """Converte [x, y, z] normalizado para (x_pixel, y_pixel)."""
    if point is None:
        return None
    return (int(point[0] * w), int(point[1] * h))


# --- FUNÇÕES DE CÁLCULO GEOMÉTRICO ---

# Função 4: Cálculo do Ângulo de Flexão/Extensão (3D)
def calculate_angle_3d(a, b, c):
    """Calcula o ângulo em graus do ponto B (vértice) entre os segmentos BA e BC."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    # Evita divisão por zero se os vetores tiverem comprimento zero
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)
    if norm_ba == 0 or norm_bc == 0:
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

    angle_deg = np.rad2deg(np.arccos(cosine_angle))
    return angle_deg


# Função 5: Cálculo de Abdução/Adução do Quadril (Rotação Lateral)
def calculate_rotation_angle(p_hip, p_knee, p_ankle, p_shoulder_oposto, is_left_side=True):
    """Estima a Abdução/Adução do quadril medindo o ângulo do fêmur com o eixo vertical da imagem."""

    # Vetor do segmento coxa (Hip -> Knee)
    segmento_coxa = np.array(p_knee) - np.array(p_hip)

    # Medir o ângulo que a coxa projeta no plano XY em relação ao eixo vertical (Y)
    segmento_coxa_xy = np.array([segmento_coxa[0], segmento_coxa[1]])

    # atan2(delta_x, -delta_y) dá o ângulo em relação ao eixo Y invertido (vertical para cima)
    angle_rad = math.atan2(segmento_coxa_xy[0], -segmento_coxa_xy[1])

    # O valor absoluto é usado para medir o deslocamento lateral a partir do vertical
    angle_deg = abs(math.degrees(angle_rad))

    return angle_deg


# --- FUNÇÃO DE ALINHAMENTO ---

def rotate_image_and_landmarks_for_analysis(image, pose_results):
    """Rotaciona a imagem e re-processa os landmarks para alinhar o quadril horizontalmente."""
    h, w, _ = image.shape
    landmarks = pose_results.pose_landmarks.landmark

    # Assumimos que a verificação de visibilidade já foi feita em perform_analysis
    hip_left = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    hip_right = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]

    x1, y1 = int(hip_left.x * w), int(hip_left.y * h)
    x2, y2 = int(hip_right.x * w), int(hip_right.y * h)

    if x2 - x1 == 0:
        angle = 90.0 if y2 > y1 else -90.0
    else:
        angle_rad = math.atan2(y2 - y1, x2 - x1)
        angle = math.degrees(angle_rad)

    correction_angle = -angle

    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, correction_angle, 1.0)

    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h),
                                   flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    # Re-processa a imagem rotacionada para obter as coordenadas corrigidas
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose_rot:
        results_rotated = pose_rot.process(cv2.cvtColor(rotated_image, cv2.COLOR_BGR2RGB))

    return rotated_image, results_rotated, correction_angle


# --- FUNÇÃO DE ANÁLISE PRINCIPAL ---

def perform_analysis(captured_image):
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        image_rgb = cv2.cvtColor(captured_image, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)

    if not results.pose_landmarks:
        return None, None, None, 0

    # 3.1. VERIFICAÇÃO INICIAL CRÍTICA: QUADRIL (PARA ALINHAMENTO)
    landmarks_all = results.pose_landmarks.landmark

    # Usamos os índices puros aqui para verificar a visibilidade antes do alinhamento
    if get_landmark_coords(landmarks_all, mp_pose.PoseLandmark.RIGHT_HIP.value) is None or \
            get_landmark_coords(landmarks_all, mp_pose.PoseLandmark.LEFT_HIP.value) is None:
        print("Pontos críticos (Quadris) não visíveis para o alinhamento. Parando a análise.")
        angles = {
            "ERRO": f"Quadris não visíveis (Menos que {VISIBILITY_THRESHOLD * 100:.0f}% de confiança). Impossível alinhar e analisar.",
            "Solução": "Tire a foto novamente, garantindo que o quadril esteja visível."
        }
        return captured_image, angles, results.pose_landmarks, 0

    # 2. ALINHAMENTO
    rotated_image, results_rotated, correction_angle = rotate_image_and_landmarks_for_analysis(captured_image, results)

    if not results_rotated.pose_landmarks:
        return None, None, None, 0

    print(f"\n--- Imagem Rotacionada em {correction_angle:.2f} graus para Alinhamento ---")

    # 3.2. EXTRAÇÃO DOS PONTOS CORRIGIDOS (COM CHECAGEM DE VISIBILIDADE)
    landmarks = results_rotated.pose_landmarks.landmark

    g = lambda index: get_landmark_coords(landmarks, index)

    # Lado Direito
    r_shoulder = g(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
    r_hip = g(mp_pose.PoseLandmark.RIGHT_HIP.value)
    r_knee = g(mp_pose.PoseLandmark.RIGHT_KNEE.value)
    r_ankle = g(mp_pose.PoseLandmark.RIGHT_ANKLE.value)
    r_heel = g(mp_pose.PoseLandmark.RIGHT_HEEL.value)
    r_elbow = g(mp_pose.PoseLandmark.RIGHT_ELBOW.value)
    r_wrist = g(mp_pose.PoseLandmark.RIGHT_WRIST.value)

    # Lado Esquerdo
    l_shoulder = g(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
    l_hip = g(mp_pose.PoseLandmark.LEFT_HIP.value)
    l_knee = g(mp_pose.PoseLandmark.LEFT_KNEE.value)
    l_ankle = g(mp_pose.PoseLandmark.LEFT_ANKLE.value)
    l_heel = g(mp_pose.PoseLandmark.LEFT_HEEL.value)
    l_elbow = g(mp_pose.PoseLandmark.LEFT_ELBOW.value)
    l_wrist = g(mp_pose.PoseLandmark.LEFT_WRIST.value)

    # PONTOS INFERIDOS
    r_costela = infer_costela_point(r_shoulder, r_hip)
    l_costela = infer_costela_point(l_shoulder, l_hip)

    # Ponto central do tronco superior (para ângulos de quadril)
    mid_shoulder = None
    if l_shoulder is not None and r_shoulder is not None:
        mid_shoulder = [(l_shoulder[0] + r_shoulder[0]) / 2, (l_shoulder[1] + r_shoulder[1]) / 2,
                        (l_shoulder[2] + r_shoulder[2]) / 2]

    # 4. CÁLCULO DE ÂNGULOS (COM CHECAGEM DE PONTOS)

    calculated_angles = {}
    nao_encontrado = "PONTO(S) NÃO ENCONTRADO(S) / VISIBILIDADE BAIXA"

    def safe_angle_calc(p1, p2, p3, angle_func=calculate_angle_3d):
        return angle_func(p1, p2, p3) if all(p is not None for p in [p1, p2, p3]) else nao_encontrado

    def safe_rot_calc(p1, p2, p3, p4, is_left=False):
        # A função calculate_rotation_angle só precisa de 3 pontos para o cálculo interno, mas 4 são usados como referência
        return calculate_rotation_angle(p1, p2, p3, p4, is_left) if all(
            p is not None for p in [p1, p2, p3]) else nao_encontrado

    # **A. MALÉOLOS E TORNOZELO**
    calculated_angles['Maléolo_D_Aprox_Tornozelo_Z'] = f"{r_ankle[2]:.4f}" if r_ankle is not None else nao_encontrado
    calculated_angles['Maléolo_E_Aprox_Tornozelo_Z'] = f"{l_ankle[2]:.4f}" if l_ankle is not None else nao_encontrado
    calculated_angles['Tornozelo_D_Flexao'] = safe_angle_calc(r_knee, r_ankle, r_heel)
    calculated_angles['Tornozelo_E_Flexao'] = safe_angle_calc(l_knee, l_ankle, l_heel)

    # **B. JOELHOS**
    calculated_angles['Joelhon_D_Flexao'] = safe_angle_calc(r_hip, r_knee, r_ankle)
    calculated_angles['Joelhon_E_Flexao'] = safe_angle_calc(l_hip, l_knee, l_ankle)

    # **C. QUADRIL (Flexão e Rotação Lateral)**
    calculated_angles['Quadril_D_Flexao'] = safe_angle_calc(mid_shoulder, r_hip, r_knee)
    calculated_angles['Quadril_E_Flexao'] = safe_angle_calc(mid_shoulder, l_hip, l_knee)
    calculated_angles['Quadril_D_Rot_Lateral'] = safe_rot_calc(r_hip, r_knee, r_ankle, l_shoulder, False)
    calculated_angles['Quadril_E_Rot_Lateral'] = safe_rot_calc(l_hip, l_knee, l_ankle, r_shoulder, True)

    # **D. REGIAO COSTELAS E TRONCO**
    calculated_angles['Costela_D_Inferencia_Y'] = f"{r_costela[1]:.4f}" if r_costela is not None else nao_encontrado
    calculated_angles['Costela_E_Inferencia_Y'] = f"{l_costela[1]:.4f}" if l_costela is not None else nao_encontrado

    if r_shoulder is not None and l_shoulder is not None and l_hip is not None:
        calculated_angles['Tronco_Inclinacao_Lateral'] = calculate_angle_3d(r_shoulder, l_shoulder, l_hip)
    else:
        calculated_angles['Tronco_Inclinacao_Lateral'] = nao_encontrado

    # **E. ACRÔMIO E OMBROS**
    calculated_angles['Acromio_D_Aprox_Ombro_Z'] = f"{r_shoulder[2]:.4f}" if r_shoulder is not None else nao_encontrado
    calculated_angles['Acromio_E_Aprox_Ombro_Z'] = f"{l_shoulder[2]:.4f}" if l_shoulder is not None else nao_encontrado
    calculated_angles['Ombro_D_Flexao'] = safe_angle_calc(r_hip, r_shoulder, r_elbow)
    calculated_angles['Ombro_E_Flexao'] = safe_angle_calc(l_hip, l_shoulder, l_elbow)

    # 5. RETORNO
    return rotated_image, calculated_angles, results_rotated.pose_landmarks, correction_angle


# --- FUNÇÃO PARA SALVAR TXT (MANTIDA) ---

def save_angles_to_txt(angles_dict, correction_angle):
    """Salva os ângulos calculados e o ângulo de alinhamento em um arquivo TXT."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"analise_pose_{timestamp}.txt"

    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=================================================\n")
            f.write(f"RELATÓRIO DE ANÁLISE DE POSE ({timestamp})\n")
            f.write("=================================================\n\n")

            f.write(f"Ângulo de Alinhamento (Correção de Inclinação da Imagem): {correction_angle:.2f} graus\n\n")

            f.write(f"LIMIAR DE VISIBILIDADE PARA PONTO VÁLIDO: {VISIBILITY_THRESHOLD * 100:.0f}%\n")
            f.write("--- ÂNGULOS DE FLEXÃO/EXTENSÃO E ROTAÇÃO (Inferidos) ---\n")

            for key, value in angles_dict.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.2f} graus\n")
                else:
                    f.write(f"{key}: {value}\n")  # Inclui a string "PONTO(S) NÃO ENCONTRADO(S)..." ou ERRO

            f.write("\n=================================================\n")
            f.write("NOTA SOBRE OS PONTOS E ROTAÇÕES:\n")
            f.write("- Acrômio, Quadril, Joelho e Tornozelo são Landmarks diretos do MediaPipe.\n")
            f.write("- Maléolo Lateral é aproximado pelo Tornozelo (ANKLE).\n")
            f.write("- Ponto 'Costela' (abaixo) é inferido geometricamente (cintura).\n")
            f.write("- Ângulos de Rotação (Rot) são uma inferência de Abdução/Adução do fêmur.\n")

        print(f"\n[SUCESSO] Relatório salvo em: {os.path.abspath(filename)}")
    except Exception as e:
        print(f"\n[ERRO] Não foi possível salvar o arquivo TXT: {e}")


# --- FUNÇÃO PRINCIPAL DE CAPTURA DA CÂMERA ---

def main_camera_capture():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Erro: Não foi possível abrir a câmera.")
        return

    print("Câmera aberta. Pressione 's' para tirar a foto, ESC (ou 'q') para sair.")

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_preview:
        captured_frame = None
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results_preview = pose_preview.process(frame_rgb)

            if results_preview.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame, results_preview.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 200, 200), thickness=2, circle_radius=2)
                )

            cv2.putText(frame, "Pressione 's' para foto, ESC/q para sair", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Camera Feed - Preview', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                captured_frame = frame.copy()
                print("Foto capturada. Processando...")
                break
            elif key == 27 or key == ord('q'):
                print("Saindo sem capturar foto.")
                break

    cap.release()
    cv2.destroyWindow('Camera Feed - Preview')

    if captured_frame is not None:
        final_image, angles, final_landmarks_result, correction_angle = perform_analysis(captured_frame)

        if final_image is not None and angles is not None:

            # Se a análise falhou na checagem do quadril (ERRO CRÍTICO)
            if "ERRO" in angles:
                save_angles_to_txt(angles, correction_angle)
                print(
                    "\n[ERRO CRÍTICO DE ANÁLISE] Falha na detecção dos quadris. Consulte o relatório TXT para detalhes.")
                return  # Sai da função após a mensagem.

            # --- FLUXO DE SUCESSO (Análise completa ou falha em pontos não críticos) ---

            save_angles_to_txt(angles, correction_angle)

            # Só desenha os landmarks se eles existirem
            if final_landmarks_result:
                mp_drawing.draw_landmarks(
                    final_image,
                    final_landmarks_result,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                )

                # Extrai os landmarks para a visualização dos pontos inferidos (Costelas)
                landmarks_viz = final_landmarks_result.landmark
                h_final, w_final, _ = final_image.shape

                get_coord = lambda index: get_landmark_coords(landmarks_viz, index)

                r_shoulder_viz = get_coord(mp_pose.PoseLandmark.RIGHT_SHOULDER.value)
                r_hip_viz = get_coord(mp_pose.PoseLandmark.RIGHT_HIP.value)
                l_shoulder_viz = get_coord(mp_pose.PoseLandmark.LEFT_SHOULDER.value)
                l_hip_viz = get_coord(mp_pose.PoseLandmark.LEFT_HIP.value)

                # Inferir o Ponto Costela
                r_costela_viz = infer_costela_point(r_shoulder_viz, r_hip_viz)
                l_costela_viz = infer_costela_point(l_shoulder_viz, l_hip_viz)

                # Desenha os pontos inferidos (Costelas) - MAGENTA
                if r_costela_viz is not None:
                    r_costela_pixel = normalize_to_pixel(r_costela_viz, w_final, h_final)
                    cv2.circle(final_image, r_costela_pixel, 5, (255, 0, 255), -1)

                if l_costela_viz is not None:
                    l_costela_pixel = normalize_to_pixel(l_costela_viz, w_final, h_final)
                    cv2.circle(final_image, l_costela_pixel, 5, (255, 0, 255), -1)

            cv2.putText(final_image, "Relatorio Salvo em TXT. Pontos Inferidos em MAGENTA.",
                        (10, final_image.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow('Pose Estimation - Aligned (Pontos em TXT)', final_image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main_camera_capture()

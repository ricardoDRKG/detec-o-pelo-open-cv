import cv2
import numpy as np


# ----------------------------------------------------------------------
# FUNÇÃO DE MÁSCARA
# ----------------------------------------------------------------------

def get_mask_hsv(image, mask_lower=(0, 100, 100), mask_upper=(15, 255, 255)):
    """
    Cria uma máscara binária no espaço de cores HSV.
    """
    # Converte a imagem BGR para HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Cria a máscara (pixels dentro do intervalo são brancos, o resto é preto)
    mask = cv2.inRange(hsv, mask_lower, mask_upper)

    return mask


# ----------------------------------------------------------------------
# NOVA FUNÇÃO: CALCULAR CENTRO
# ----------------------------------------------------------------------

def calcular_centro(contour):
    """Calcula o centróide (centro de massa) de um contorno."""
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        return (cX, cY)
    return None


# Inicializa a webcam
webcam = cv2.VideoCapture(0)

# Variável para armazenar o último frame ANTES de sair
last_original_frame = None

if webcam.isOpened():
    validacao, frame = webcam.read()

    # Se a câmera abriu, entra no loop de captura
    while validacao:
        # 1. Captura o frame atual
        validacao, frame = webcam.read()

        if not validacao:
            break

        # Armazena o frame original para salvar depois
        last_original_frame = frame.copy()

        # 2. Gera a máscara
        # OBS: Ajuste a faixa de cores (0, 183, 170) e (124, 255, 255)
        # para corresponder aos objetos que você quer rastrear na sua imagem.
        mask = get_mask_hsv(frame, mask_lower=(0, 183, 170), mask_upper=(124, 255, 255))

        # 3. Aplica a máscara ao frame (Mostra APENAS a cor detectada)
        frame_mascarado = cv2.bitwise_and(frame, frame, mask=mask)

        # -------------------------------------------------------------------
        # NOVO CÓDIGO: CONTORNOS E CONEXÃO
        # -------------------------------------------------------------------

        # Encontra os contornos na máscara pura
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        centros_encontrados = []

        # Desenha todos os contornos encontrados (opcional)
        cv2.drawContours(frame_mascarado, contours, -1, (255, 0, 0), 2)  # Azul

        # Itera sobre os contornos para calcular seus centros
        for contour in contours:
            # Filtro por área mínima para ignorar ruídos (ajuste o valor 500 conforme necessário)
            if cv2.contourArea(contour) > 500:
                centro = calcular_centro(contour)
                if centro:
                    centros_encontrados.append(centro)
                    # Desenha um ponto vermelho no centro
                    cv2.circle(frame_mascarado, centro, 5, (0, 0, 255), -1)

                    # Se houver pelo menos DOIS objetos, traça a linha entre o primeiro e o segundo
        if len(centros_encontrados) >= 2:
            ponto_A = centros_encontrados[0]
            ponto_B = centros_encontrados[1]

            cor_conexao = (0, 255, 0)  # Verde
            espessura_conexao = 3

            # Traça a linha entre os dois centros na janela Mascarada
            cv2.line(frame_mascarado, ponto_A, ponto_B, cor_conexao, espessura_conexao)

        # -------------------------------------------------------------------
        # FIM DO NOVO CÓDIGO
        # -------------------------------------------------------------------

        # 4. Exibe em três janelas SEPARADAS:
        cv2.imshow("1. Video Original (Saia com ESC)", frame)
        cv2.imshow("2. Mascara Aplicada + Conexao", frame_mascarado)  # Janela com a linha
        cv2.imshow("3. Mascara Pura", mask)

        # 5. Verifica a tecla de saída
        key = cv2.waitKey(5)
        if key == 27:  # 27 é o código ASCII para a tecla ESC
            print("ESC pressionado. Salvando o último frame original.")
            break

    # 6. Salva o frame capturado
    img_name = "Fotobraba.png"
    # Salva a ÚLTIMA máscara gerada (que contém o resultado binário)
    if frame_mascarado is not None:
        cv2.imwrite(img_name, frame_mascarado)
        print(f"Máscara salva como {img_name}")
    else:
        print("Nenhum frame válido foi capturado para salvar.")

# 7. Limpeza final
webcam.release()
cv2.destroyAllWindows()
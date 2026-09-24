import cv2
import numpy as np

# Inicia a captura de vídeo pela webcam padrão (índice 0).
cap = cv2.VideoCapture(0)

while True:
    # Lê cada frame da webcam. 'ret' indica sucesso/insucesso e 'frame' é a imagem capturada.
    ret, frame = cap.read()
    if not ret:
        break

    # Converte o frame do espaço de cor BGR (padrão do OpenCV) para HSV.
    # HSV (Hue, Saturation, Value) facilita a segmentação de cores.
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define o intervalo de cor azul no espaço HSV.
    # Estes valores podem variar consoante a iluminação e a tonalidade pretendida.
    # Hue (110 a 130) -> faixa de azuis.
    # Saturation e Value ajustam a sensibilidade.
    lower_blue = np.array([110, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Cria a máscara binária onde os pixéis dentro do intervalo de cor azul são brancos (255)
    # e os demais são pretos (0).
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Aplica uma operação bitwise-AND entre o frame original e ele próprio, utilizando a máscara.
    # Isto mantém apenas as regiões onde a máscara é branca, resultando nas áreas azuis.
    res = cv2.bitwise_and(frame, frame, mask=mask)

    # Mostra as janelas:
    # "Frame Original" - imagem original da webcam.
    # "Mascara" - mostra a máscara binária.
    # "Resultado" - mostra apenas a parte azul da imagem, "removendo" o resto.
    cv2.imshow('Frame Original', frame)
    cv2.imshow('Mascara', mask)
    cv2.imshow('Resultado', res)

    # Espera 5ms por uma tecla.
    # Se a tecla ESC (código 27) for pressionada, sai do loop.
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

# Liberta a webcam e fecha todas as janelas.
cap.release()
cv2.destroyAllWindows()

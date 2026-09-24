import numpy as np
import cv2

# Inicializar a captura de vídeo
cap = cv2.VideoCapture('vtest.avi')
# Se quiser usar a webcam, descomente a linha abaixo e comente a anterior
# cap = cv2.VideoCapture(0)

# Escolher método de subtração de fundo
method = 1

if method == 0:
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
elif method == 1:
    bgSubtractor = cv2.createBackgroundSubtractorMOG2()
else:
    bgSubtractor = cv2.bgsegm.createBackgroundSubtractorGMG()

# Criar um kernel para operações morfológicas
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

while True:
    # Ler o próximo frame do vídeo
    ret, frame = cap.read()
    if not ret:
        break

    # Início da contagem de tempo
    tick_start = cv2.getTickCount()

    # Aplicar a subtração de fundo
    foregroundMask = None #<--- PARA MUDAR

    # Remover ruídos com operação morfológica de abertura
    foregroundMask = cv2.morphologyEx(foregroundMask, cv2.MORPH_OPEN, kernel)

    # Fim da contagem de tempo
    tick_end = cv2.getTickCount()

    # Cálculo do tempo de processamento do frame
    time_per_frame = (tick_end - tick_start) / cv2.getTickFrequency()

    # Exibir a máscara resultante
    cv2.imshow('background subtraction', foregroundMask)

    # Mostrar opcionalmente o tempo de processamento no console
    # (Caso não deseje imprimir, pode comentar a linha abaixo)
    print(f"Tempo por frame: {time_per_frame:.6f} segundos")

    # Esperar 30 ms por uma tecla; se for ESC (27), sair do loop
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

# Liberar recursos
cap.release()
cv2.destroyAllWindows()

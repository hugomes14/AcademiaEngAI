import numpy as np
import cv2

# Capturar vídeo a partir de ficheiro
cap = cv2.VideoCapture('vtest.avi')

# Converter o espaço de cores de BGR para tons de cinzento
# Mostrar o fotograma resultante
# Premir 'q' para sair
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imshow('frame', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Se preferir, usar um atraso maior:
    # if cv2.waitKey(25) & 0xFF == ord('q'):
    #     break

# Libertar todos os recursos após a conclusão
cap.release()
cv2.destroyAllWindows()

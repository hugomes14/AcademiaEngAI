import numpy as np
import cv2

# Capturar vídeo da câmara
cap = cv2.VideoCapture(0)

# Definir resolução 720p
cap.set(3, 1280)
cap.set(4, 720)

# OU definir resolução 480p
#cap.set(3, 640)
#cap.set(4, 480)

# Converter o espaço de cores de BGR para RGB
# Mostrar o fotograma resultante
# Premir 'q' para sair
while(True):

    # Capturar fotograma a fotograma
    ret, frame = cap.read()

    # As operações sobre o fotograma são feitas aqui
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

    # Mostrar o fotograma resultante
    #cv2.imshow('frame',gray)
    cv2.imshow('frame',color)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Quando tudo estiver concluído, libertar a captura
cap.release()
cv2.destroyAllWindows()

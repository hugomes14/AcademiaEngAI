import numpy as np
import cv2

# Capturar vídeo da câmara


# Definir resolução 720p


# OU definir resolução 480p
#cap.set(3, 640)
#cap.set(4, 480)

# Converter o espaço de cores de BGR para RGB
# Mostrar o fotograma resultante
# Premir 'q' para sair
while(True):

    # Capturar fotograma a fotograma


    # As operações sobre o fotograma são feitas aqui


    # Mostrar o fotograma resultante


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Quando tudo estiver concluído, libertar a captura

cv2.destroyAllWindows()

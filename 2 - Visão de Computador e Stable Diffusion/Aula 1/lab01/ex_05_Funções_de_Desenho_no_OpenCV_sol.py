import numpy as np
import cv2

# Criar uma imagem preta de 512x512 com 3 canais (BGR)
img = np.zeros((512,512,3), np.uint8)

# Desenhar uma linha diagonal azul com espessura de 5 pixels
img = cv2.line(img, (0,0), (511,511), (255,0,0), 5)

# Desenhar um retângulo verde com espessura de 3 pixels
img = cv2.rectangle(img, (384,0), (510,128), (0,255,0), 3)

# Desenhar um círculo vermelho preenchido
img = cv2.circle(img, (447,63), 63, (0,0,255), -1)

# Desenhar uma elipse preenchida
img = cv2.ellipse(img, (256,256), (100,50), 0, 0, 180, 255, -1)

# Desenhar um polígono
pts = np.array([[10,5],[20,30],[70,20],[50,10]], np.int32)
pts = pts.reshape((-1,1,2))
img = cv2.polylines(img, [pts], True, (0,255,255))

# Adicionar texto "Viseu"
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'Viseu', (10,500), font, 4, (255,255,255), 2, cv2.LINE_AA)

# Exibir a imagem
cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

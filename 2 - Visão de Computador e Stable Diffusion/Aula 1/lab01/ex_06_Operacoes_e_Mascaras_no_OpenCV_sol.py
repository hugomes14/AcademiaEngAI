import cv2
import numpy as np

# Arrays uint8 para demonstração das operações aritméticas
x = np.uint8([[250]])
y = np.uint8([[10]])

print("Soma com cv2.add:", cv2.add(x, y))  # Soma saturada (250+10=260, saturado em 255)
print("Soma com NumPy  :", x + y)         # Soma modular (250+10=260 % 256=4)

# Carregar a imagem de fundo (Ovar.jpg)
img1 = cv2.imread('Ovar.jpg')
if img1 is None:
    raise IOError("Erro ao carregar 'Ovar.jpg'. Verifique o nome e a localização do arquivo.")

# Carregar a imagem do robô (bot.png)
img2 = cv2.imread('bot.png')
if img2 is None:
    raise IOError("Erro ao carregar 'bot.png'. Verifique o nome e a localização do arquivo.")

# Definir a ROI na imagem de fundo baseada no tamanho do robô
rows, cols, channels = img2.shape
roi = img1[0:rows, 0:cols]

# Converter a imagem do robô para escala de cinza e criar máscara
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
mask_inv = cv2.bitwise_not(mask)

# Remover a área do robô da ROI do fundo
img1_bg = cv2.bitwise_and(roi, roi, mask=mask_inv)

# Obter apenas a área do robô
img2_fg = cv2.bitwise_and(img2, img2, mask=mask)

# Combinar as áreas do fundo e do robô
dst = cv2.add(img1_bg, img2_fg)
img1[0:rows, 0:cols] = dst

# Carregar a terceira imagem de fundo (bot_background.png)
img3 = cv2.imread('bot_background.png')
if img3 is None:
    raise IOError("Erro ao carregar 'bot_background.png'. Verifique o nome e a localização do arquivo.")

# Redimensionar img3 para ter as mesmas dimensões de img1
img3 = cv2.resize(img3, (img1.shape[1], img1.shape[0]))

# Realizar o blending entre img1 (com o robô) e img3
final = cv2.addWeighted(img1, 0.7, img3, 0.3, 0)

# Exibir as imagens
cv2.imshow('Imagem com Robo', img1)
cv2.imshow('Imagem de Fundo Extra', img3)
cv2.imshow('Resultado Blending', final)
cv2.waitKey(0)
cv2.destroyAllWindows()

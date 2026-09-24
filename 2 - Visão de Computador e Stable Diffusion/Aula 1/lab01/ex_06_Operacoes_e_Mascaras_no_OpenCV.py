import cv2
import numpy as np

# Arrays uint8 para demonstração das operações aritméticas
x = np.uint8([[250]])
y = np.uint8([[10]])

# print("Soma com cv2.add:", )  # Soma saturada (250+10=260, saturado em 255)
# print("Soma com NumPy  :", x + y)         # Soma modular (250+10=260 % 256=4)

# Carregar a imagem de fundo (Ovar.jpg)
img1 = cv2.imread('Ovar.jpg')
if img1 is None:
    raise IOError("Erro ao carregar 'Ovar.jpg'. Verifique o nome e a localização do arquivo.")

# Carregar a imagem do robô (bot.png)
img2 = cv2.imread('bot.png')
if img2 is None:
    raise IOError("Erro ao carregar 'bot.png'. Verifique o nome e a localização do arquivo.")

# Definir a ROI na imagem de fundo baseada no tamanho do robô


# Converter a imagem do robô para escala de cinza e criar máscara


# Remover a área do robô da ROI do fundo


# Obter apenas a área do robô


# Combinar as áreas do fundo e do robô


# Carregar a terceira imagem de fundo (bot_background.png)
img3 = cv2.imread('bot_background.png')
if img3 is None:
    raise IOError("Erro ao carregar 'bot_background.png'. Verifique o nome e a localização do arquivo.")

# Redimensionar img3 para ter as mesmas dimensões de img1


# Realizar o blending entre img1 (com o robô) e img3
# final =

# Exibir as imagens
cv2.imshow('Imagem com Robo', img1)
cv2.imshow('Imagem de Fundo Extra', img3)
# cv2.imshow('Resultado Blending', final)        # <--------- Descomentar no fim
cv2.waitKey(0)
cv2.destroyAllWindows()

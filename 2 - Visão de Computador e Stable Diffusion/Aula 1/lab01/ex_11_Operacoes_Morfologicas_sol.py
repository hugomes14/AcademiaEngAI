import cv2
import numpy as np

# Carrega a imagem "bot.png" em escala de cinza.
# O segundo parâmetro 0 indica que a imagem será carregada em tons de cinza.
img = cv2.imread('bot.png', 0)
if img is None:
    raise IOError("Erro ao carregar 'bot.png'. Verifique o nome e a localização do arquivo.")

# Cria um kernel (elemento estruturante) de tamanho 5x5.
# np.ones((5,5), np.uint8) cria uma matriz 5x5 de uns, com tipo de dados unsigned int 8 bits.
kernel = np.ones((5,5), np.uint8)

# Erosão (cv2.erode)
# cv2.erode(imagem, kernel, iterations) erode a imagem "afinando" os objetos.
# iterations=1 significa que a operação é realizada uma única vez.
erosion = cv2.erode(img, kernel, iterations=1)

# Dilatação (cv2.dilate)
# cv2.dilate(imagem, kernel, iterations) dilata a imagem "engrossando" os objetos.
dilation = cv2.dilate(img, kernel, iterations=1)

# Abertura (cv2.morphologyEx com MORPH_OPEN)
# Equivale a uma erosão seguida de dilatação.
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

# Fechamento (cv2.morphologyEx com MORPH_CLOSE)
# Equivale a uma dilatação seguida de erosão.
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

# Gradiente Morfológico (cv2.morphologyEx com MORPH_GRADIENT)
# É a diferença entre dilatação e erosão, destacando contornos.
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)

# Top-Hat (cv2.morphologyEx com MORPH_TOPHAT)
# É a diferença entre a imagem original e a sua abertura,
# destacando áreas mais claras do que o fundo.
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)

# Black-Hat (cv2.morphologyEx com MORPH_BLACKHAT)
# É a diferença entre o fechamento da imagem e a imagem original,
# destacando áreas mais escuras do que o fundo.
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)

# Exibe as imagens resultantes em janelas separadas.
# cv2.imshow('nome_da_janela', imagem) mostra a imagem em uma janela.
cv2.imshow('Original', img)
cv2.imshow('Erosion', erosion)
cv2.imshow('Dilation', dilation)
cv2.imshow('Opening', opening)
cv2.imshow('Closing', closing)
cv2.imshow('Gradient', gradient)
cv2.imshow('Top-Hat', tophat)
cv2.imshow('Black-Hat', blackhat)

# cv2.waitKey(0) aguarda até que uma tecla seja pressionada para prosseguir.
cv2.waitKey(0)

# cv2.destroyAllWindows() fecha todas as janelas abertas pelo OpenCV.
cv2.destroyAllWindows()

import cv2
import numpy as np

# Carrega a imagem "bot.png" em escala de cinza.
# O segundo parâmetro 0 indica que a imagem será carregada em tons de cinza.
img = cv2.imread('bot.png', 0)
if img is None:
    raise IOError("Erro ao carregar 'bot.png'. Verifique o nome e a localização do arquivo.")

# Cria um kernel (elemento estruturante) de tamanho 5x5.
# np.ones((5,5), np.uint8) cria uma matriz 5x5 de uns, com tipo de dados unsigned int 8 bits.
kernel = None

# Erosão (cv2.erode)
# cv2.erode(imagem, kernel, iterations) erode a imagem "afinando" os objetos.
# iterations=1 significa que a operação é realizada uma única vez.
erosion = None

# Dilatação (cv2.dilate)
# cv2.dilate(imagem, kernel, iterations) dilata a imagem "engrossando" os objetos.
dilation = None

# Abertura (cv2.morphologyEx com MORPH_OPEN)
# Equivale a uma erosão seguida de dilatação.
opening = None

# Fechamento (cv2.morphologyEx com MORPH_CLOSE)
# Equivale a uma dilatação seguida de erosão.
closing = None

# Gradiente Morfológico (cv2.morphologyEx com MORPH_GRADIENT)
# É a diferença entre dilatação e erosão, destacando contornos.
gradient = None

# Top-Hat (cv2.morphologyEx com MORPH_TOPHAT)
# É a diferença entre a imagem original e a sua abertura,
# destacando áreas mais claras do que o fundo.
tophat = None

# Black-Hat (cv2.morphologyEx com MORPH_BLACKHAT)
# É a diferença entre o fechamento da imagem e a imagem original,
# destacando áreas mais escuras do que o fundo.
blackhat = None

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

import cv2
import numpy as np

# Carrega a imagem "bot.png"
# A função cv2.imread(nome_do_arquivo, modo) carrega a imagem.
# Se não especificarmos o modo, o padrão é carregar a imagem em BGR (modo = 1).
img = cv2.imread('bot.png')
if img is None:
    raise IOError("Erro ao carregar 'bot.png'. Verifique o nome e a localização do arquivo.")

# Aplica o filtro de média (cv2.blur)
# cv2.blur(imagem, (ksize_x, ksize_y)) -> aplica um filtro de média com um kernel de tamanho 5x5.
blur = cv2.blur(img, (5,5))

# Aplica o filtro Gaussiano (cv2.GaussianBlur)
# cv2.GaussianBlur(imagem, (ksize_x, ksize_y), sigmaX)
# (ksize_x, ksize_y) devem ser valores ímpares. sigmaX = desvio padrão no eixo X; 
# se for 0, o OpenCV calcula automaticamente.
gaussian_blur = cv2.GaussianBlur(img, (5,5), 0)

# Aplica o filtro Mediana (cv2.medianBlur)
# cv2.medianBlur(imagem, ksize) -> ksize é um valor ímpar que define o tamanho do kernel.
# Esse filtro substitui cada píxel pela mediana dos pixeis vizinhos.
median = cv2.medianBlur(img, 5)

# Aplica o filtro Bilateral (cv2.bilateralFilter)
# cv2.bilateralFilter(imagem, d, sigmaColor, sigmaSpace)
# d = diâmetro do kernel; sigmaColor = desvio padrão no espaço da cor;
# sigmaSpace = desvio padrão no espaço das coordenadas da imagem.
# Esse filtro suaviza a imagem, mas preserva as arestas.
bilateral = cv2.bilateralFilter(img, 9, 75, 75)

# Exibe as imagens em janelas separadas
# cv2.imshow('nome_da_janela', imagem) mostra a imagem em uma janela.
cv2.imshow('Original', img)
cv2.imshow('Blur (Média)', blur)
cv2.imshow('Gaussian Blur', gaussian_blur)
cv2.imshow('Median Blur', median)
cv2.imshow('Bilateral Filter', bilateral)

# A função cv2.waitKey(0) faz o programa esperar até que uma tecla seja pressionada.
cv2.waitKey(0)

# cv2.destroyAllWindows() fecha todas as janelas abertas pelo OpenCV.
cv2.destroyAllWindows()

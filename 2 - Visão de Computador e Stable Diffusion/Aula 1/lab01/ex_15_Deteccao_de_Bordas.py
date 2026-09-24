import cv2
import numpy as np

# Carrega a imagem "Ovar.jpg" em tons de cinza.
# O segundo parâmetro 0 indica que a imagem será carregada em escala de cinza.
img = cv2.imread('Ovar.jpg', 0)
if img is None:
    raise IOError("Erro ao carregar 'Ovar.jpg'. Verifique o nome e a localização do arquivo.")

# Detecção de Bordas com Canny
# cv2.Canny(imagem, threshold1, threshold2) -> threshold1 e threshold2 definem os limites do histéreses.
# A detecção Canny segue vários passos internos para destacar bordas, resultando em bordas bem definidas.
edges_canny = None

# Detecção de Bordas com Sobel
# Sobel é um operador de gradiente que calcula a derivada da imagem separadamente em x e em y.
# cv2.Sobel(imagem, depth, dx, dy, ksize) -> dx, dy indicam a direção da derivada; ksize é o tamanho do kernel.
  # Derivada em x
  # Derivada em y

# Calcula a magnitude do gradiente resultante da combinação de sobelx e sobely.
# cv2.magnitude(x, y) calcula a raiz quadrada (x^2 + y^2).
sobel_magnitude = None

# Convertendo a magnitude para valores de 8 bits para exibição.
# cv2.convertScaleAbs converte a escala de uma matriz para 8 bits absolutos, ideal para exibir.


# Detecção de Bordas com Laplacian
# O operador Laplaciano calcula a segunda derivada da imagem, destacando mudanças abruptas de intensidade.
laplacian = None

# Convertendo a escala do resultado do Laplacian para exibição (8 bits).


# Exibindo as imagens
# cv2.imshow('nome_da_janela', imagem) abre uma janela com a imagem.
cv2.imshow('Original', img)
cv2.imshow('Canny', edges_canny)
cv2.imshow('Sobel Magnitude', sobel_magnitude)
cv2.imshow('Laplacian', laplacian)

# Aguarda uma tecla ser pressionada para fechar as janelas.
cv2.waitKey(0)
cv2.destroyAllWindows()

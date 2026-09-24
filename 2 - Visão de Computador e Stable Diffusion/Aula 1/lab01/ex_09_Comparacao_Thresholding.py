import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregar a imagem em tons de cinza
img = cv2.imread('sudoku.png', 0)
if img is None:
    raise IOError("Erro ao carregar 'sudoku.png'. Verifique o nome e a localização do arquivo.")

# Aplicar filtro mediana para redução de ruído


# Limiarização Global (Simple Thresholding)


# Limiarização Adaptativa (Média)


# Limiarização Adaptativa (Gaussiana)


# Exibir resultados com Matplotlib
titles = [
    'Original Image',
    'Global Thresholding (v = 127)',
    'Adaptive Mean Thresholding',
    'Adaptive Gaussian Thresholding'
]

images = [img, th1, th2, th3]

for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

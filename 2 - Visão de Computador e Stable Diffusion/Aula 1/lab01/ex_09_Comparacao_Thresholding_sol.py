import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carregar a imagem em tons de cinza
img = cv2.imread('sudoku.png', 0)
if img is None:
    raise IOError("Erro ao carregar 'sudoku.png'. Verifique o nome e a localização do arquivo.")

# Aplicar filtro mediana para redução de ruído
img = cv2.medianBlur(img, 5)

# Limiarização Global (Simple Thresholding)
ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

# Limiarização Adaptativa (Média)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                            cv2.THRESH_BINARY, 11, 2)

# Limiarização Adaptativa (Gaussiana)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv2.THRESH_BINARY, 11, 2)

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

import cv2 as cv
from matplotlib import pyplot as plt
import numpy as np
import time

# Fonte de dados
left_image = cv.imread('assets/left.png', cv.IMREAD_GRAYSCALE)
right_image = cv.imread('assets/right.png', cv.IMREAD_GRAYSCALE)

# Quanto maior o bloco, mais suave (e potencialmente menos precisa) será a estimativa de disparidade
stereo = cv.StereoBM_create(numDisparities=0, blockSize=21)
depth = stereo.compute(left_image, right_image)

print(depth)

plt.figure(figsize=(16, 4))
plt.suptitle('Estimativa de Profundidade a partir de Imagens Binoculares')

ax = plt.subplot(1, 3, 1)
img_title = 'Imagem Esquerda'
image = plt.imread('assets/left.png')
plt.imshow(image, cmap=plt.cm.binary)
plt.title(img_title, fontsize='small')
plt.axis(False)

ax = plt.subplot(1, 3, 2)
img_title = 'Imagem Direita'
image = plt.imread('assets/right.png')
plt.imshow(image, cmap=plt.cm.binary)
plt.title(img_title, fontsize='small')
plt.axis(False)

ax = plt.subplot(1, 3, 3)
img_title = 'Mapa de Profundidade'
plt.imshow(depth)
plt.title(img_title, fontsize='small')
plt.axis(False)

plt.show()

import cv2
import numpy as np

# Carregar a imagem Ovar.jpg
img = cv2.imread('Ovar.jpg')
if img is None:
    raise IOError("Erro ao carregar 'Ovar.jpg'. Verifique o nome e a localização do arquivo.")

height, width = img.shape[:2]

# Escalamento (Resize)
# Aumentar 2x
# Reduzir 0.5x

# PyrDown e PyrUp
# Reduz a imagem usando pirâmides Gaussianas
# Aumenta a imagem usando pirâmides Gaussianas

# Translação (Translation)
# Deslocar 100 pixeis à direita e 50 pixeis para baixo


# Rotação (Rotation) 45º

# Transformação Afim (Affine Transform)
pts1 = np.float32([[50, 50],
                   [200, 50],
                   [50, 200]])
pts2 = np.float32([[10, 100],
                   [200, 50],
                   [100, 250]])


# Transformação por Perspectiva (Perspective Transform)
pts1_persp = np.float32([[50,50],
                         [width-50,50],
                         [50,height-50],
                         [width-50,height-50]])
pts2_persp = np.float32([[0,0],
                         [300,0],
                         [0,300],
                         [300,300]])


# Exibir as imagens
# cv2.imshow('Original', img)
# cv2.imshow('Scale Up (2x)', img_scale_up)
# cv2.imshow('Scale Down (0.5x)', img_scale_down)
# cv2.imshow('PyrDown', img_pyr_down)
# cv2.imshow('PyrUp', img_pyr_up)
# cv2.imshow('Translation', img_translation)
# cv2.imshow('Rotation 45 graus', img_rotation)
# cv2.imshow('Affine Transformation', img_affine)
# cv2.imshow('Perspective Transformation', img_perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()

import cv2
import numpy as np

# Carregar a imagem Ovar.jpg
img = cv2.imread('Ovar.jpg')
if img is None:
    raise IOError("Erro ao carregar 'Ovar.jpg'. Verifique o nome e a localização do arquivo.")

height, width = img.shape[:2]

# Escalamento (Resize)
img_scale_up = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)    # Aumentar 2x
img_scale_down = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA) # Reduzir 0.5x

# PyrDown e PyrUp
img_pyr_down = cv2.pyrDown(img)   # Reduz a imagem usando pirâmides Gaussianas
img_pyr_up = cv2.pyrUp(img)       # Aumenta a imagem usando pirâmides Gaussianas

# Translação (Translation)
tx, ty = 100, 50  # Deslocar 100 pixeis à direita e 50 pixeis para baixo
M_translation = np.float32([[1, 0, tx],
                            [0, 1, ty]])
img_translation = cv2.warpAffine(img, M_translation, (width, height))

# Rotação (Rotation)
center = (width//2, height//2)
angle = 45
scale = 1.0
M_rotation = cv2.getRotationMatrix2D(center, angle, scale)
img_rotation = cv2.warpAffine(img, M_rotation, (width, height))

# Transformação Afim (Affine Transform)
pts1 = np.float32([[50, 50],
                   [200, 50],
                   [50, 200]])
pts2 = np.float32([[10, 100],
                   [200, 50],
                   [100, 250]])
M_affine = cv2.getAffineTransform(pts1, pts2)
img_affine = cv2.warpAffine(img, M_affine, (width, height))

# Transformação por Perspectiva (Perspective Transform)
pts1_persp = np.float32([[50,50],
                         [width-50,50],
                         [50,height-50],
                         [width-50,height-50]])
pts2_persp = np.float32([[0,0],
                         [300,0],
                         [0,300],
                         [300,300]])
M_perspective = cv2.getPerspectiveTransform(pts1_persp, pts2_persp)
img_perspective = cv2.warpPerspective(img, M_perspective, (300, 300))

# Exibir as imagens
cv2.imshow('Original', img)
cv2.imshow('Scale Up (2x)', img_scale_up)
cv2.imshow('Scale Down (0.5x)', img_scale_down)
cv2.imshow('PyrDown', img_pyr_down)
cv2.imshow('PyrUp', img_pyr_up)
cv2.imshow('Translation', img_translation)
cv2.imshow('Rotation 45 graus', img_rotation)
cv2.imshow('Affine Transformation', img_affine)
cv2.imshow('Perspective Transformation', img_perspective)

cv2.waitKey(0)
cv2.destroyAllWindows()

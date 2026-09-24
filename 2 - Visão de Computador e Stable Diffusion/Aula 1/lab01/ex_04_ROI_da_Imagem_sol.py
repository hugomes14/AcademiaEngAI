import cv2
import numpy as np

# Carregar a imagem
img = cv2.imread('Ovar.jpg')
if img is None:
    print("Erro ao carregar a imagem.")
    exit()

# Informações sobre a imagem
print(f"Shape da imagem: {img.shape}")  # Altura, largura e canais
print(f"Tamanho da imagem: {img.size} pixels")  # Total de pixels
print(f"Tipo da imagem: {img.dtype}")  # Tipo de dados dos pixels

# Definir uma Região de Interesse (ROI)
ROI = img[300:400, 400:600]

# Substituir uma área da imagem pela ROI
img[0:100, 0:200] = ROI

# Exibir a imagem modificada
cv2.imshow("Imagem com ROI Modificada", img)

# Aguardar uma tecla para fechar
cv2.waitKey(0)
cv2.destroyAllWindows()

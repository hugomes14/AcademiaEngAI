import numpy as np  # Biblioteca para manipulação de arrays
import cv2  # Biblioteca OpenCV para processamento de imagens

img = cv2.imread('ovar.jpg', cv2.IMREAD_GRAYSCALE)  # Carregar a imagem em tons de cinza

if img is None:
    print("Erro: Não foi possível carregar a imagem. Verifique o nome do ficheiro e a localização.")
    exit()

cv2.imwrite('ovar.png', img)  # Salva a imagem como um ficheiro PNG
print("Imagem gravada com sucesso como 'ovar.png'.")
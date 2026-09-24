import numpy as np  # Biblioteca para manipulação de arrays
import cv2  # Biblioteca OpenCV para processamento de imagens

img = cv2.imread('Ovar.jpg', cv2.IMREAD_COLOR)  # Carrega a imagem 'Ovar.jpg' a cores (BGR)

if img is None:
    print("Erro: Não foi possível carregar a imagem. Verifique o nome do ficheiro e a localização.")
    exit()

cv2.imshow('imagem', img)  # Exibe a imagem numa janela com o nome 'imagem'
cv2.waitKey(0)  # Pausa o programa até que o utilizador pressione qualquer tecla
cv2.destroyAllWindows()  # Fecha todas as janelas abertas
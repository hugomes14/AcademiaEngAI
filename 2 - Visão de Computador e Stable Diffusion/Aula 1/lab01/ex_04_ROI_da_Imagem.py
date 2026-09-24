import cv2
import numpy as np

# Carregar a imagem
img = cv2.imread('Ovar.jpg')
if img is None:
    print("Erro ao carregar a imagem.")
    exit()

# Informações sobre a imagem
# Altura, largura e canais
# Total de pixels
# Tipo de dados dos pixels

# Definir uma Região de Interesse (ROI) - desde (400, 300) até (600,400)


# Substituir uma área da imagem pela ROI em (0, 0) até (200,100)


# Exibir a imagem modificada


# Aguardar uma tecla para fechar
cv2.waitKey(0)
cv2.destroyAllWindows()
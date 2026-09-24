import cv2
import numpy as np
from matplotlib import pyplot as plt

# Carrega a imagem "Ovar.jpg" em tons de cinza
# O segundo parâmetro '0' indica a leitura em escala de cinza.
gray_img = cv2.imread('Ovar.jpg', 0)
if gray_img is None:
    raise IOError("Erro ao carregar 'Ovar.jpg'. Verifique o nome e a localização do arquivo.")

# Redimensiona a imagem para 1024x768 para melhor visualização.
# cv2.resize(img, (largura, altura))
imS = cv2.resize(gray_img, (1024, 768))

# Exibe a imagem redimensionada em uma janela.
cv2.imshow('Imagem Ovar Redimensionada', imS)

# Calcula o histograma da imagem com 256 bins.
# cv2.calcHist([imagem], [canal], mascara, [numero_de_bins], [intervalo_de_intensidade])
# Aqui:
#   [gray_img] -> imagem de entrada
#   [0] -> canal 0 (em tons de cinza só existe um canal)
#   None -> sem máscara, o histograma será de toda a imagem
#   [256] -> número de bins do histograma
#   [0,256] -> intervalo das intensidades (0 a 255)
hist_256 = None
print('hist_256',hist_256)

# Cria uma figura do Matplotlib para exibir o histograma
plt.figure(figsize=(8,4))

# Exibe o histograma com 256 bins
plt.subplot(1,2,1)
# plt.hist(array_de_pixels, numero_de_bins, intervalo)
# gray_img.ravel() -> transforma a imagem em um array 1D de pixels.
plt.hist(gray_img.ravel(), 256, [0,256])
plt.title('Histograma (256 bins)')
plt.xlabel('Intensidade')
plt.ylabel('Frequência')

# Calcula o histograma com 64 bins para comparação
hist_64 = None
print('hist_64',hist_64)

# Exibe o histograma com 64 bins no mesmo plot, ao lado
plt.subplot(1,2,2)
plt.hist(gray_img.ravel(), 64, [0,256])
plt.title('Histograma (64 bins)')
plt.xlabel('Intensidade')
plt.ylabel('Frequência')

plt.tight_layout()
plt.show()

# Aguarda a tecla ESC ser pressionada para fechar a janela da imagem.
while True:
    k = cv2.waitKey(0) & 0xFF     
    if k == 27: # ESC
        break

cv2.destroyAllWindows()

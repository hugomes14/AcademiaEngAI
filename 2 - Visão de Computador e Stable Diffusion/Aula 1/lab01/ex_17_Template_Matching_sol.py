import cv2
import numpy as np

# Carrega a imagem original (coins_euros.jpg), que contém várias moedas.
# Se a imagem não for encontrada, lança um erro.
img_rgb = cv2.imread('coins_euros.jpg')
if img_rgb is None:
    raise IOError("Erro ao carregar 'coins_euros.jpg'. Verifique o nome e a localização do arquivo.")

# Converte a imagem original para tons de cinza, pois o template matching
# costuma ser aplicado em imagens em escala de cinza para simplificar o processo.
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

# Carrega a imagem do template (coin_20cents.jpg), que é a moeda que desejamos encontrar.
# Também é carregada em tons de cinza (parâmetro 0).
template = cv2.imread('coin_20cents.jpg', 0)
if template is None:
    raise IOError("Erro ao carregar 'coin_20cents.jpg'. Verifique o nome e a localização do arquivo.")

# Obtém as dimensões do template.
# template.shape retorna (altura, largura).
# template.shape[::-1] inverte a tupla, resultando em (largura, altura).
w, h = template.shape[::-1]

# Aplica o template matching para encontrar regiões na imagem original
# que sejam semelhantes ao template.
# cv2.matchTemplate(imagem, template, método) retorna um mapa de correspondências.
# cv2.TM_CCOEFF_NORMED é um método que retorna valores normalizados entre -1 e 1,
# onde valores próximos de 1 indicam alta semelhança.
res = cv2.matchTemplate(img_gray, template, cv2.TM_CCOEFF_NORMED)

# Define um limiar (threshold) para considerar uma correspondência como válida.
# Quanto mais próximo de 1, mais estrita a correspondência.
threshold = 0.8

# np.where(res >= threshold) retorna as coordenadas (y, x) onde o valor em 'res'
# é maior ou igual ao limiar escolhido.
loc = np.where(res >= threshold)

# Para cada ponto (pt) encontrado, desenha um retângulo na imagem original.
# zip(*loc[::-1]) inverte a ordem de loc e desempacota as coordenadas para (x, y).
for pt in zip(*loc[::-1]):
    # Desenha um retângulo vermelho com espessura de 2 pixels.
    # pt é o canto superior esquerdo do retângulo,
    # (pt[0] + w, pt[1] + h) é o canto inferior direito.
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,0,255), 2)

# Salva a imagem resultante com as detecções marcadas.
cv2.imwrite('res_euros.jpg', img_rgb)

# Opcionalmente, exibe a imagem com as detecções em uma janela do OpenCV.
cv2.imshow('Detecções', img_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()

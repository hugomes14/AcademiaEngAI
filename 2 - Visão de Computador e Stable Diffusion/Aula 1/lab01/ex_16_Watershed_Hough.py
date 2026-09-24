import cv2
import numpy as np

# Carrega a imagem "bot.png" em BGR.
# Caso não encontre a imagem, gera um erro.
img = cv2.imread('bot.png')
if img is None:
    raise IOError("Erro ao carregar 'bot.png'. Verifique o nome e a localização do arquivo.")

# ----------------------------------------------------------------
# Segmentação por Watershed
# ----------------------------------------------------------------

# Converter a imagem para tons de cinza, pois várias operações morfológicas
# e thresholding são aplicadas em escala de cinza.


# Aplica um threshold usando Otsu para segmentar foreground e background.
# cv2.threshold(imagem, limiar, max_val, tipo)
# THRESH_BINARY_INV -> inverte a binarização (objetos claros viram pretos, fundo vira branco)
# + OTSU -> encontra o melhor limiar automaticamente.


# Remover ruídos pequenos com abertura (erosão seguida de dilatação).
# cv2.morphologyEx(imagem, op, kernel) aplica operações morfológicas.


# Dilata a imagem para obter a área de fundo "certa" (sure_bg).


# Distance transform para encontrar a área de foreground "certa" (sure_fg).
# cv2.distanceTransform calcula a distância de cada píxel até o zero mais próximo.


# Converte sure_fg para uint8, necessário para connectedComponents.


# unknown = áreas que não sabemos se são fg ou bg.


# Obtém marcadores a partir dos componentes conectados do foreground certo.
# Cada componente recebe um ID.


# Ajusta os marcadores adicionando 1, assim o background será 1 e não 0.


# Pixels desconhecidos marcados como 0.


# Aplica o algoritmo Watershed.
# cv2.watershed modifica 'markers', colocando -1 onde há bordas entre regiões.


# Cria uma cópia da imagem e marca as fronteiras detectadas pelo Watershed (-1) em vermelho.


# ----------------------------------------------------------------
# Detecção de Linhas (Hough Lines)
# ----------------------------------------------------------------

# Converter para escala de cinza para detecção de bordas.


# Detecção de bordas com Canny, preparando para a Transformada de Hough.


# cv2.HoughLinesP encontra linhas usando a transformada de Hough probabilística.
# parâmetros:
#   1, np.pi/180 -> resolução do espaço de parâmetros (rho=1 pixel, theta=1 grau)
#   100 -> limiar mínimo de votos
#   minLineLength=100 -> comprimento mínimo da linha
#   maxLineGap=10 -> gap máximo entre segmentos para serem considerados uma mesma linha


        # Desenha a linha na imagem com cor verde e espessura 2.


# ----------------------------------------------------------------
# Detecção de Círculos (Hough Circles)
# ----------------------------------------------------------------

# Converter para escala de cinza e suavizar com mediana.


# cv2.HoughCircles usa a transformada de Hough para detecção de círculos.
# Parâmetros principais:
#   dp=1 -> resolução do acumulador
#   minDist=50 -> distância mínima entre centros de círculos detectados
#   param1=100 e param2=30 -> limiares internos do algoritmo
#   minRadius, maxRadius = 0 -> Detecta qualquer raio.



        # Desenha o círculo detectado em azul.

        # Desenha o centro do círculo em vermelho.


# ----------------------------------------------------------------
# Exibir resultados
# ----------------------------------------------------------------

cv2.imshow('Original', img)
cv2.imshow('Watershed Result', watershed_result)
cv2.imshow('Hough Lines', hough_lines_result)
cv2.imshow('Hough Circles', hough_circles_result)

cv2.waitKey(0)
cv2.destroyAllWindows()

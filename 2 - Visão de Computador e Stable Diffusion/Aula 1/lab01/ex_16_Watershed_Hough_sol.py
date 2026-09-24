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
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Aplica um threshold usando Otsu para segmentar foreground e background.
# cv2.threshold(imagem, limiar, max_val, tipo)
# THRESH_BINARY_INV -> inverte a binarização (objetos claros viram pretos, fundo vira branco)
# + OTSU -> encontra o melhor limiar automaticamente.
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

# Remover ruídos pequenos com abertura (erosão seguida de dilatação).
# cv2.morphologyEx(imagem, op, kernel) aplica operações morfológicas.
kernel = np.ones((3,3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

# Dilata a imagem para obter a área de fundo "certa" (sure_bg).
sure_bg = cv2.dilate(opening, kernel, iterations=3)

# Distance transform para encontrar a área de foreground "certa" (sure_fg).
# cv2.distanceTransform calcula a distância de cada píxel até o zero mais próximo.
dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
ret, sure_fg = cv2.threshold(dist_transform, 0.7*dist_transform.max(), 255, 0)

# Converte sure_fg para uint8, necessário para connectedComponents.
sure_fg = np.uint8(sure_fg)

# unknown = áreas que não sabemos se são fg ou bg.
unknown = cv2.subtract(sure_bg, sure_fg)

# Obtém marcadores a partir dos componentes conectados do foreground certo.
# Cada componente recebe um ID.
ret, markers = cv2.connectedComponents(sure_fg)

# Ajusta os marcadores adicionando 1, assim o background será 1 e não 0.
markers = markers + 1

# Pixels desconhecidos marcados como 0.
markers[unknown == 255] = 0

# Aplica o algoritmo Watershed.
# cv2.watershed modifica 'markers', colocando -1 onde há bordas entre regiões.
markers = cv2.watershed(img, markers)

# Cria uma cópia da imagem e marca as fronteiras detectadas pelo Watershed (-1) em vermelho.
watershed_result = img.copy()
watershed_result[markers == -1] = [0, 0, 255]

# ----------------------------------------------------------------
# Detecção de Linhas (Hough Lines)
# ----------------------------------------------------------------

# Converter para escala de cinza para detecção de bordas.
gray_for_lines = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detecção de bordas com Canny, preparando para a Transformada de Hough.
edges = cv2.Canny(gray_for_lines, 50, 150, apertureSize=3)

# cv2.HoughLinesP encontra linhas usando a transformada de Hough probabilística.
# parâmetros:
#   1, np.pi/180 -> resolução do espaço de parâmetros (rho=1 pixel, theta=1 grau)
#   100 -> limiar mínimo de votos
#   minLineLength=100 -> comprimento mínimo da linha
#   maxLineGap=10 -> gap máximo entre segmentos para serem considerados uma mesma linha
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=100, maxLineGap=10)
hough_lines_result = img.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # Desenha a linha na imagem com cor verde e espessura 2.
        cv2.line(hough_lines_result, (x1,y1), (x2,y2), (0,255,0), 2)

# ----------------------------------------------------------------
# Detecção de Círculos (Hough Circles)
# ----------------------------------------------------------------

# Converter para escala de cinza e suavizar com mediana.
gray_for_circles = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_for_circles = cv2.medianBlur(gray_for_circles, 5)

# cv2.HoughCircles usa a transformada de Hough para detecção de círculos.
# Parâmetros principais:
#   dp=1 -> resolução do acumulador
#   minDist=50 -> distância mínima entre centros de círculos detectados
#   param1=100 e param2=30 -> limiares internos do algoritmo
#   minRadius, maxRadius = 0 -> Detecta qualquer raio.
circles = cv2.HoughCircles(gray_for_circles, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                           param1=100, param2=30, minRadius=0, maxRadius=0)

hough_circles_result = img.copy()

if circles is not None:
    circles = np.uint16(np.around(circles))
    for c in circles[0,:]:
        # Desenha o círculo detectado em azul.
        cv2.circle(hough_circles_result, (c[0], c[1]), c[2], (255,0,0), 2)
        # Desenha o centro do círculo em vermelho.
        cv2.circle(hough_circles_result, (c[0], c[1]), 2, (0,0,255), 3)

# ----------------------------------------------------------------
# Exibir resultados
# ----------------------------------------------------------------

cv2.imshow('Original', img)
cv2.imshow('Watershed Result', watershed_result)
cv2.imshow('Hough Lines', hough_lines_result)
cv2.imshow('Hough Circles', hough_circles_result)

cv2.waitKey(0)
cv2.destroyAllWindows()

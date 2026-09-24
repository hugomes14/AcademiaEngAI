import cv2
import numpy as np

# Carrega a imagem "Ovar.jpg" em BGR.
# Caso a imagem não seja encontrada, o programa gera um erro.
img = cv2.imread('Ovar.jpg')
if img is None:
    raise IOError("Erro ao carregar 'Ovar.jpg'. Verifique o nome e a localização do arquivo.")

# Cria uma máscara do mesmo tamanho da imagem, inicializada com zeros.
# Esta máscara é usada pelo algoritmo GrabCut para separar fundo e primeiro plano.
mask = np.zeros(img.shape[:2], np.uint8)

# Cria os modelos de fundo (bgdModel) e de primeiro plano (fgdModel).
# São arrays requeridos pelo GrabCut para manter informações internas durante o processo.
# Eles devem ter o tamanho (1,65) e tipo float64.
bgdModel = np.zeros((1,65), np.float64)
fgdModel = np.zeros((1,65), np.float64)

# Define um retângulo que engloba a área aproximada do objeto a ser extraído.
# Formato: (x, y, largura, altura).
# Ajuste conforme a imagem para garantir que o objeto de interesse fique dentro deste retângulo.
rect = (50, 50, 450, 290)

# Aplica o algoritmo GrabCut.
# cv2.grabCut(imagem, máscara, retângulo, bgdModel, fgdModel, iterações, modo)
# Modo: cv2.GC_INIT_WITH_RECT -> inicializa a partir do retângulo definido.
# 5 iterações costumam ser suficientes para um bom resultado inicial.


# Após o GrabCut, a máscara contém:
# 0 e 2 -> Píxeis considerados fundo.
# 1 e 3 -> Píxeis considerados primeiro plano (foreground).
# Aqui, convertemos todos os píxeis 0 e 2 em 0 (fundo) e 1 e 3 em 1 (primeiro plano).


# Multiplica a imagem pelos valores da máscara, mantendo apenas o primeiro plano.
# Onde mask2 = 1, a imagem é mantida, onde mask2 = 0, fica preto.
img_result = None

# Exibe a imagem original e a imagem resultante.
# Aqui recarregamos a imagem original para exibir sem modificações.
cv2.imshow('Original', cv2.imread('Ovar.jpg'))
cv2.imshow('Foreground Extraido', img_result)

# Espera uma tecla ser pressionada para fechar as janelas.
cv2.waitKey(0)
cv2.destroyAllWindows()

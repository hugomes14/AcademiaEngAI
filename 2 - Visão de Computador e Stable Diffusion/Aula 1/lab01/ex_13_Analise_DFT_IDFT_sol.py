import cv2
import numpy as np

# Carrega a imagem "Ovar.jpg" em escala de cinza.
# O segundo argumento 0 indica que será carregada como imagem em tons de cinza.
img = cv2.imread('Ovar.jpg', 0)
if img is None:
    raise IOError("Erro ao carregar 'Ovar.jpg'. Verifique o nome e a localização do arquivo.")

# Converter a imagem para float32, pois a DFT requer esse tipo.
# np.float32 converte os valores de intensidade para floats de 32 bits.
dft_input = np.float32(img)

# Aplicar a DFT (Transformada Discreta de Fourier).
# cv2.dft(src, flags) -> src: entrada, flags: parâmetros adicionais.
# cv2.DFT_COMPLEX_OUTPUT faz com que a saída tenha 2 canais: parte real e imaginária.
dft = cv2.dft(dft_input, flags=cv2.DFT_COMPLEX_OUTPUT)

# Realiza o shift do espectro para colocar as frequências baixas no centro.
# np.fft.fftshift move a origem do espectro para o centro da imagem.
dft_shift = np.fft.fftshift(dft)

# Calcula o espectro de magnitude:
# A magnitude é sqrt(re^2 + im^2), e muitas vezes aplicamos log(1+...) para melhorar a visualização.
# cv2.magnitude(x, y) calcula a magnitude para cada ponto (re, im).
magnitude = cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1])
magnitude_spectrum = 20 * np.log(magnitude + 1)

# Normaliza a magnitude para o intervalo [0, 255] para melhor exibição.
# cv2.normalize(src, dst, alfa, beta, tipo_normalização) -> NORM_MINMAX normaliza para intervalo [alfa, beta].
magnitude_spectrum = cv2.normalize(magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX)
magnitude_spectrum = np.uint8(magnitude_spectrum)

# Exibe a imagem original e o espectro de magnitude.
cv2.imshow('Imagem Original', img)
cv2.imshow('Espectro de Magnitude', magnitude_spectrum)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Para reconstruir a imagem, primeiro desfazemos o shift.
# np.fft.ifftshift move de volta as frequências baixas para o canto.
dft_ishift = np.fft.ifftshift(dft_shift)

# Aplica a IDFT (Transformada Inversa de Fourier).
# cv2.idft() retorna a parte complexa no domínio espacial novamente.
img_back_complex = cv2.idft(dft_ishift)

# Extrair apenas a magnitude resultante, que corresponde aos valores reais da imagem reconstruída.
img_back = cv2.magnitude(img_back_complex[:,:,0], img_back_complex[:,:,1])

# Normaliza a imagem reconstruída para o intervalo [0,255] para exibição.
img_back_normalized = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)
img_back_normalized = np.uint8(img_back_normalized)

# Exibe a imagem reconstruída.
cv2.imshow('Imagem Reconstruida (IDFT)', img_back_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()

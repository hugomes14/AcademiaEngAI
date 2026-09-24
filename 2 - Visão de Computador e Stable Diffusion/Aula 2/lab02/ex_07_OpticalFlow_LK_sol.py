import numpy as np
import cv2

# Inicializa a captura de vídeo a partir do ficheiro "vtest.avi".
# Caso pretenda usar a webcam, comente a linha abaixo e descomente a seguinte.
cap = cv2.VideoCapture('vtest.avi')
# cap = cv2.VideoCapture(0)

# Parâmetros para detecção de cantos com Shi-Tomasi
feature_params = dict(maxCorners=100,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parâmetros para o Optical Flow de Lucas-Kanade
lk_params = dict(winSize=(15, 15),
                 maxLevel=2,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Gera cores aleatórias para desenhar as trajetórias
color = np.random.randint(0, 255, (100, 3))

# Ler o primeiro frame do vídeo
ret, old_frame = cap.read()
if not ret:
    print("Não foi possível ler o primeiro frame do vídeo.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Converter o primeiro frame para tons de cinza
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

# Detetar pontos/cantos iniciais na primeira frame
p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Criar uma máscara (imagem preta) para desenhar as trajetórias
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Medição do tempo no início do processamento do frame
    tick_start = cv2.getTickCount()

    # Converter o frame atual para cinza
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Calcular o fluxo óptico entre o frame anterior e o atual
    # p1 são os novos pontos correspondentes a p0
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Selecionar apenas os pontos com bom resultado (st == 1)
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Desenhar as linhas e pontos do fluxo óptico
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        # Converter coordenadas para inteiro
        a, b, c, d = map(int, [a, b, c, d])
        mask = cv2.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv2.circle(frame, (a, b), 5, color[i].tolist(), -1)

    # Combinar a frame atual com a máscara das trajetórias
    img = cv2.add(frame, mask)

    # Medição do tempo no fim do processamento do frame
    tick_end = cv2.getTickCount()
    time_per_frame = (tick_end - tick_start) / cv2.getTickFrequency()
    # Opcional: imprimir o tempo no terminal
    print(f"Tempo por frame: {time_per_frame:.6f} segundos")

    # Mostrar a frame com as trajetórias
    cv2.imshow('frame', img)
    k = cv2.waitKey(30) & 0xff
    if k == 27:  # Tecla ESC
        break

    # Atualizar o frame e os pontos para a próxima iteração
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cv2.destroyAllWindows()
cap.release()

import numpy as np
import cv2

# Iniciar a captura da webcam
cap = cv2.VideoCapture(0)

# Ler o primeiro frame da webcam
ret, frame = cap.read()
if not ret:
    print("Não foi possível ler o primeiro frame da webcam.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

rows, cols = frame.shape[:2]

# Definir as dimensões da janela (ROI) no centro do frame
windowWidth = 150
windowHeight = 200
windowCol = int((cols - windowWidth) / 2)
windowRow = int((rows - windowHeight) / 2)
window = (windowCol, windowRow, windowWidth, windowHeight)

# Extrair a ROI e converter para HSV
roi = frame[windowRow:windowRow + windowHeight, windowCol:windowCol + windowWidth]
roiHsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

# Definir limites para ignorar áreas escuras
lowLimit = np.array((0., 60., 32.))
highLimit = np.array((180., 255., 255.))
mask = cv2.inRange(roiHsv, lowLimit, highLimit)

# Calcular e normalizar o histograma da ROI
roiHist = cv2.calcHist([roiHsv], [0], mask, [180], [0, 180])
cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)

# Critérios de paragem do CamShift: 10 iterações ou movimento < 1 píxel
terminationCriteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Início da medição de tempo
    tick_start = cv2.getTickCount()

    # Converter o frame atual para HSV
    frameHsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Backprojection do histograma na imagem atual
    backprojectedFrame = cv2.calcBackProject([frameHsv], [0], roiHist, [0, 180], 1)

    # Aplicar a mesma máscara para ignorar áreas escuras
    mask = cv2.inRange(frameHsv, lowLimit, highLimit)
    backprojectedFrame &= mask

    # Aplicar o CamShift para encontrar o novo retângulo rotacionado
    rotated_rect, window = cv2.CamShift(backprojectedFrame, window, terminationCriteria)

    # Desenhar o polígono resultante do CamShift
    points = cv2.boxPoints(rotated_rect)
    points = points.astype(int)
    frame = cv2.polylines(frame, [points], True, (255, 0, 0), 2)

    # Fim da medição de tempo
    tick_end = cv2.getTickCount()
    time_per_frame = (tick_end - tick_start) / cv2.getTickFrequency()

    # Mostrar o tempo por frame (opcional)
    print(f"Tempo por frame: {time_per_frame:.6f} segundos")

    cv2.imshow('CamShift Tracking', frame)
    k = cv2.waitKey(60) & 0xFF
    if k == 27:  # Tecla ESC para sair
        break

# Libertar recursos
cap.release()
cv2.destroyAllWindows()

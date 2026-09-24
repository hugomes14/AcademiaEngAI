import numpy as np
import cv2

# Carrega os classificadores Haar Cascade para rosto e olhos.
# Certifique-se de que os ficheiros "haarcascade_frontalface_default.xml" e "haarcascade_eye.xml" 
# estejam dentro da pasta "haarcascades" ou no caminho indicado.
faceCascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
eyeCascade = cv2.CascadeClassifier('haarcascades/haarcascade_eye.xml')

# Inicia a captura de vídeo a partir da webcam (índice 0)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Início da contagem de tempo para medir desempenho
    tick_start = cv2.getTickCount()

    # Converter o frame para tons de cinza, pois a deteção Haar Cascade funciona melhor a P&B
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detetar rostos na imagem
    # scaleFactor = 1.3: Reduz a imagem em 30% para procurar rostos em diferentes escalas
    # minNeighbors = 5: Indica quantos vizinhos, no mínimo, cada deteção deve ter para ser considerada válida
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for x, y, w, h in faces:
        # Desenhar um retângulo em torno do rosto detetado
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Definir a região de interesse (ROI) para a deteção dos olhos dentro do rosto
        roiGray = gray[y:y+h, x:x+w]

        # Detetar olhos na região do rosto
        eyes = eyeCascade.detectMultiScale(roiGray)

        # Limitar a deteção a dois olhos por rosto
        eyeCounter = 0
        for ex, ey, ew, eh in eyes:
            if eyeCounter > 1:
                break
            cv2.rectangle(frame, (x + ex, y + ey), (x + ex + ew, y + ey + eh), (0, 255, 0), 2)
            eyeCounter += 1

    # Fim da contagem de tempo para este frame
    tick_end = cv2.getTickCount()
    time_per_frame = (tick_end - tick_start) / cv2.getTickFrequency()

    # Imprimir o tempo por frame no terminal (opcional)
    print(f"Tempo por frame: {time_per_frame:.6f} segundos")

    cv2.imshow('frame', frame)

    # Pressionar 'q' para sair
    k = cv2.waitKey(20) & 0xFF
    if k == ord('q'):
        break

# Libertar recursos e fechar janelas
cap.release()
cv2.destroyAllWindows()

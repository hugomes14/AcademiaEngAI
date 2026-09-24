import numpy as np
import cv2

# Carrega os classificadores Haar Cascade para rosto e olhos.
# Certifique-se de que os ficheiros "haarcascade_frontalface_default.xml" e "haarcascade_eye.xml" 
# estejam dentro da pasta "haarcascades" ou no caminho indicado.


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



        # Desenhar um retângulo em torno do rosto detetado


        # Definir a região de interesse (ROI) para a deteção dos olhos dentro do rosto


        # Detetar olhos na região do rosto


        # Limitar a deteção a dois olhos por rosto


    # Fim da contagem de tempo para este frame


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

import cv2
# Necessário instalar a biblioteca ultralytics, por exemplo: pip install ultralytics
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Carregar o modelo de deteção YOLO (sem segmentação)
model = YOLO("yolo11n.pt")
# model = YOLO("best.pt")

# Captura de vídeo a partir da webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(2, 720)


# Obter as dimensões e fps (frames por segundo) do vídeo de entrada
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))

# Criar um escritor de vídeo para guardar o resultado do processamento
out = cv2.VideoWriter("object-detection-tracking.avi",
                      cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

# Ciclo principal para ler e processar cada fotograma do vídeo
while True:
    ret, im0 = cap.read()
    if not ret:
        # Se não for possível ler mais fotogramas, sair do ciclo
        break

    # Inicializar o objeto 'annotator' para desenhar caixas e anotações no fotograma
    annotator = Annotator(im0, line_width=2)

    # Executar o tracking (acompanhamento) de objetos no fotograma atual
    results = model.track(im0, persist=True)

    # Se existirem deteções com IDs, desenhar as caixas delimitadoras no fotograma
    if results[0].boxes.id is not None:
        boxes = results[0].boxes.xyxy.cpu()
        track_ids = results[0].boxes.id.int().cpu().tolist()
        classes = results[0].boxes.cls.int().cpu().tolist()

        # Para cada objeto detetado, desenhar a caixa, cor e etiqueta (classe + ID)
        for box, track_id, cls in zip(boxes, track_ids, classes):
            color = colors(track_id, True)
            label = f"{model.names[cls]} {track_id}"
            annotator.box_label(box, label, color=color)

    # Escrever o fotograma anotado no vídeo de saída
    out.write(im0)
    # Mostrar o fotograma anotado numa janela
    cv2.imshow("object-detection-tracking", im0)

    # Se o utilizador carregar na tecla 'q', sair do ciclo
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libertar o escritor de vídeo e o objeto de captura de vídeo
out.release()
cap.release()
# Fechar todas as janelas abertas
cv2.destroyAllWindows()
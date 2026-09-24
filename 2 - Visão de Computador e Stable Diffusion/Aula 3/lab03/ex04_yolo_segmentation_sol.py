import cv2
import numpy as np 
# Necessário instalar a biblioteca ultralytics, por exemplo: pip install ultralytics
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Carregar o modelo de segmentação YOLO
model = YOLO("yolo11n-seg.pt")
# model = YOLO("best.pt")

# Captura de vídeo a partir de um ficheiro (ou webcam id 0)
cap = cv2.VideoCapture(0)

cap.set(3, 1280)
cap.set(2, 720)


# Obter as dimensões e fps
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH,
                                       cv2.CAP_PROP_FRAME_HEIGHT,
                                       cv2.CAP_PROP_FPS))

# Criar um escritor de vídeo
out = cv2.VideoWriter("instance-segmentation-object-tracking.avi",
                      cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

while True:
    ret, im0 = cap.read()
    if not ret:
        break

    # Inicializar o annotator
    annotator = Annotator(im0, line_width=2)

    # Executar o tracking
    results = model.track(im0, persist=True)

    # Verificar se existem deteções, IDs e máscaras
    if results[0].boxes.id is not None and results[0].masks is not None:
        # Obter máscaras, IDs e também as Caixas (Boxes)
        masks = results[0].masks.xy
        track_ids = results[0].boxes.id.int().cpu().tolist()
        boxes = results[0].boxes.xyxy.cpu() # <--- Necessário para desenhar a etiqueta

        # Ciclo atualizado para incluir 'box'
        for box, mask, track_id in zip(boxes, masks, track_ids):
            color = colors(int(track_id), True)
            txt_color = annotator.get_txt_color(color)

            # 1. Desenhar a Máscara (Polígono)
            # Convertemos a máscara para int32 para o OpenCV desenhar
            cv2.fillPoly(im0, [np.array(mask, dtype=np.int32)], color)

            # 2. Desenhar a Caixa e o ID (Usando o método oficial)
            annotator.box_label(box, label=str(track_id), color=color, txt_color=txt_color)

    # Escrever o fotograma anotado no vídeo de saída
    out.write(im0)
    # Mostrar o fotograma anotado numa janela
    cv2.imshow("instance-segmentation-object-tracking", im0)

    # Se o utilizador carregar na tecla 'q', sair do ciclo
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Libertar o escritor de vídeo e o objeto de captura de vídeo
out.release()
cap.release()
# Fechar todas as janelas abertas
cv2.destroyAllWindows()
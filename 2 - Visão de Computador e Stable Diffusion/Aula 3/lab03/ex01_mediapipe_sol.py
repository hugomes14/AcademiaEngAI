# Instalar as dependencias necessárias
# pip install mediapipe (python 3.12)

import argparse

import cv2
import mediapipe as mp

parser = argparse.ArgumentParser(description='Deteção Objectron a partir da webcam ou de um vídeo.')
parser.add_argument('video', nargs='?', help='Ficheiro de vídeo para usar no lugar da webcam')
args = parser.parse_args()

if not hasattr(cv2, 'VideoCapture'):
    raise RuntimeError('OpenCV está incompleto neste ambiente. Reinstale opencv-contrib-python no venv.')

# Inicialização das ferramentas de desenho e do modelo Objectron
mp_drawing = mp.solutions.drawing_utils
mp_objectron = mp.solutions.objectron

# Captura de vídeo a partir da webcam (índice 0) ou de um ficheiro
cap = cv2.VideoCapture(args.video if args.video else 0)
if not cap.isOpened():
    raise RuntimeError(
        f'Não foi possível abrir {args.video or "a webcam (índice 0)"}. '
        'No WSL2, confirme a existência de /dev/video0 ou indique um ficheiro de vídeo.'
    )

# Criação do contexto do Objectron com parâmetros específicos:
# - Modo de imagem estática desativado (static_image_mode=False),
#   o que significa que o Objectron assume input contínuo (vídeo)
# - max_num_objects define o número máximo de objetos a detetar
# - min_detection_confidence e min_tracking_confidence definem a confiança mínima
#   para deteção e acompanhamento
# - model_name define o tipo de objeto a detetar (neste caso, 'Cup')
with mp_objectron.Objectron(static_image_mode=False,
                            max_num_objects=5,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.99,
                            model_name='Cup') as objectron:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      # Parar quando terminar o vídeo ou a câmara deixar de fornecer imagens
      print('Não foi possível ler outro fotograma.')
      break

    # Para melhorar o desempenho, opcionalmente marcar a imagem como não modificável
    # (writeable=False), permitindo que o processamento seja feito por referência.
    image.flags.writeable = False
    # Converter a imagem de BGR (OpenCV) para RGB (MediaPipe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processar a imagem com o Objectron
    results = objectron.process(image)

    # Preparar a imagem para desenho de anotações
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Se forem detetados objetos, desenhar as landmarks dos objetos e o eixo 3D
    if results.detected_objects:
        for detected_object in results.detected_objects:
            mp_drawing.draw_landmarks(
                image, 
                detected_object.landmarks_2d, 
                mp_objectron.BOX_CONNECTIONS
            )
            mp_drawing.draw_axis(
                image, 
                detected_object.rotation, 
                detected_object.translation
            )

    # Espelhar a imagem horizontalmente para visualização estilo "selfie"
    cv2.imshow('MediaPipe Objectron', cv2.flip(image, 1))

    # Se a tecla ESC (código 27) for pressionada, sair do loop
    if cv2.waitKey(5) & 0xFF == 27:
      break

# Libertar a captura de vídeo
cap.release()
cv2.destroyAllWindows()

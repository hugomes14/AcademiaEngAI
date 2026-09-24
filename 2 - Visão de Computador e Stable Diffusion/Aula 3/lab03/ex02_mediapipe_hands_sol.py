# Faz o seguinte:
# Executa o código após instalar as dependências necessárias, por exemplo:
# pip install mediapipe

import cv2
import mediapipe as mp

# Inicialização das ferramentas de desenho e do módulo de deteção de mãos do MediaPipe
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Captura de vídeo a partir da webcam (índice 0)
cap = cv2.VideoCapture(0)

# Configurar o contexto 'Hands' do MediaPipe:
# - model_complexity: nível de complexidade do modelo (0 é o mais simples)
# - min_detection_confidence: confiança mínima para considerar que a deteção da mão é válida
# - min_tracking_confidence: confiança mínima para o acompanhamento dos pontos da mão
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

  while cap.isOpened():
    success, image = cap.read()
    if not success:
      # Se não for possível ler um fotograma da câmara (por exemplo, está vazio),
      # ignorar este caso e continuar.
      # Caso estivesse a ler de um ficheiro de vídeo, poder-se-ia usar 'break'.
      print("A ignorar fotograma vazio da câmara.")
      continue

    # Para melhorar o desempenho, opcionalmente marcar a imagem como não modificável
    # (writeable=False), permitindo que o processamento seja feito por referência.
    image.flags.writeable = False
    # Converter a imagem de BGR (usado pelo OpenCV) para RGB (usado pelo MediaPipe)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Processar a imagem com o modelo de deteção de mãos
    results = hands.process(image)

    # Preparar a imagem para desenhar as anotações
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Se forem detetadas mãos, desenhar os marcadores (landmarks) e as conexões
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())

    # Espelhar a imagem horizontalmente para um efeito "selfie"
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    # Se o utilizador pressionar a tecla ESC (27), sair do loop
    if cv2.waitKey(5) & 0xFF == 27:
      break

# Libertar a captura de vídeo e fechar as janelas
cap.release()
cv2.destroyAllWindows()

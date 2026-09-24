# Faz o seguinte:
# Executa o código após instalar as dependências necessárias, por exemplo:
# pip install mediapipe

import cv2
import mediapipe as mp

# Inicialização das ferramentas de desenho e do módulo de deteção de mãos do MediaPipe


# Captura de vídeo a partir da webcam (índice 0)


# Configurar o contexto 'Hands' do MediaPipe:
# - model_complexity: nível de complexidade do modelo (0 é o mais simples)
# - min_detection_confidence: confiança mínima para considerar que a deteção da mão é válida
# - min_tracking_confidence: confiança mínima para o acompanhamento dos pontos da mão


      # Se não for possível ler um fotograma da câmara (por exemplo, está vazio),
      # ignorar este caso e continuar.
      # Caso estivesse a ler de um ficheiro de vídeo, poder-se-ia usar 'break'.

    # Para melhorar o desempenho, opcionalmente marcar a imagem como não modificável
    # (writeable=False), permitindo que o processamento seja feito por referência.


    # Converter a imagem de BGR (usado pelo OpenCV) para RGB (usado pelo MediaPipe)


    # Processar a imagem com o modelo de deteção de mãos


    # Preparar a imagem para desenhar as anotações


    # Se forem detetadas mãos, desenhar os marcadores (landmarks) e as conexões


    # Espelhar a imagem horizontalmente para um efeito "selfie"


    # Se o utilizador pressionar a tecla ESC (27), sair do loop


# Libertar a captura de vídeo e fechar as janelas


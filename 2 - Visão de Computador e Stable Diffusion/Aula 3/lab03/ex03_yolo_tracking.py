import cv2
# Necessário instalar a biblioteca ultralytics, por exemplo: pip install ultralytics


# Carregar o modelo de deteção YOLO (sem segmentação)


# Captura de vídeo a partir da webcam



# Obter as dimensões e fps (frames por segundo) do vídeo de entrada


# Criar um escritor de vídeo para guardar o resultado do processamento


# Ciclo principal para ler e processar cada fotograma do vídeo

        # Se não for possível ler mais fotogramas, sair do ciclo


    # Inicializar o objeto 'annotator' para desenhar caixas e anotações no fotograma


    # Executar o tracking (acompanhamento) de objetos no fotograma atual


    # Se existirem deteções com IDs, desenhar as caixas delimitadoras no fotograma


        # Para cada objeto detetado, desenhar a caixa, cor e etiqueta (classe + ID)


    # Escrever o fotograma anotado no vídeo de saída

    # Mostrar o fotograma anotado numa janela


    # Se o utilizador carregar na tecla 'q', sair do ciclo


# Libertar o escritor de vídeo e o objeto de captura de vídeo

# Fechar todas as janelas abertas

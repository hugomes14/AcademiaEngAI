# Instalar as dependencias necessárias
# pip install mediapipe (python 3.12)



# Inicialização das ferramentas de desenho e do modelo Objectron


# Captura de vídeo a partir da webcam (índice 0)


# Criação do contexto do Objectron com parâmetros específicos:
# - Modo de imagem estática desativado (static_image_mode=False),
#   o que significa que o Objectron assume input contínuo (vídeo)
# - max_num_objects define o número máximo de objetos a detetar
# - min_detection_confidence e min_tracking_confidence definem a confiança mínima
#   para deteção e acompanhamento
# - model_name define o tipo de objeto a detetar (neste caso, 'Cup')




      # Se não for possível ler um fotograma da câmara, ignorar ou continuar
      # Para um vídeo gravado, seria mais apropriado usar 'break'

    # Para melhorar o desempenho, opcionalmente marcar a imagem como não modificável
    # (writeable=False), permitindo que o processamento seja feito por referência.


    # Converter a imagem de BGR (OpenCV) para RGB (MediaPipe)


    # Processar a imagem com o Objectron


    # Preparar a imagem para desenho de anotações


    # Se forem detetados objetos, desenhar as landmarks dos objetos e o eixo 3D


    # Espelhar a imagem horizontalmente para visualização estilo "selfie"

    # Se a tecla ESC (código 27) for pressionada, sair do loop

# Libertar a captura de vídeo


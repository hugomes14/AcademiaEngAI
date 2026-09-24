import cv2
import numpy as np 
# Necessário instalar a biblioteca ultralytics, por exemplo: pip install ultralytics


# Carregar o modelo de segmentação YOLO


# Captura de vídeo a partir de um ficheiro (ou webcam id 0)


# Obter as dimensões e fps


# Criar um escritor de vídeo




    # Inicializar o annotator


    # Executar o tracking


    # Verificar se existem deteções, IDs e máscaras

        # Obter máscaras, IDs e também as Caixas (Boxes)
        
        
                                                         # <--- Necessário para desenhar a etiqueta

        # Ciclo atualizado para incluir 'box'


            # 1. Desenhar a Máscara (Polígono)
            # Convertemos a máscara para int32 para o OpenCV desenhar


            # 2. Desenhar a Caixa e o ID (Usando o método oficial)


    # Escrever o fotograma anotado no vídeo de saída

    # Mostrar o fotograma anotado numa janela


    # Se o utilizador carregar na tecla 'q', sair do ciclo


# Libertar o escritor de vídeo e o objeto de captura de vídeo

# Fechar todas as janelas abertas

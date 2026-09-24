# pip install bs4

import cv2
import datetime
import requests
from bs4 import BeautifulSoup
import numpy as np

def obter_temperatura(cidade):
    # Substitui espaços por '+' para a pesquisa no Google
    cidade_query = cidade.replace(" ", "+")
    URL = f"https://www.google.com/search?q=temperatura+em+{cidade_query}"
    headers = {
        "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/114.0.0.0 Safari/537.36")
    }

    response = requests.get(URL, headers=headers)
    soup = BeautifulSoup(response.text, "html.parser")

    try:
        # Normalmente, a temperatura encontra-se numa tag <span> com classe "wob_t"
        temperatura = soup.find("span", class_="wob_t").text
        return temperatura
    except AttributeError:
        return None

nome = "Ricardo Pinto"
CITY = "Viseu"

temperatura_str = obter_temperatura(CITY)
if temperatura_str is None:
    temperatura_str = "N/A"

# Inicia a captura de vídeo através da webcam (índice 0)
cap = cv2.VideoCapture(0)

# Define a resolução para 1280x720 (opcional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    print("Não foi possível aceder à webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Falha ao capturar o frame de vídeo.")
        break

    # Ajusta ligeiramente o contraste e o brilho da imagem da webcam
    # alpha é o factor de contraste, beta o brilho


    # Obtém as dimensões do frame


    # Definições para o texto (cor vermelha, espessura 2, fonte, etc.)
    color = (0, 0, 255)  # Vermelho em BGR
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    margin = 20

    # Obtém a data e hora atuais
    agora = datetime.datetime.now()
    data_hora = agora.strftime("%d/%m/%Y %H:%M:%S")

    # Texto da temperatura
    if temperatura_str != "N/A":
        temp_text = f"Temp: {temperatura_str}C"
    else:
        temp_text = "Temp: N/A"

    # Desenha um retângulo em torno do frame (opcional)


    # Canto superior esquerdo: Nome da Pessoa


    # Canto superior direito: Data e Hora


    # Canto inferior esquerdo: Cidade


    # Canto inferior direito: Temperatura


    # Exibe o vídeo numa janela intitulada "Eu (aqui)"
    cv2.imshow("Eu (aqui)", frame)

    # Pressione 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# -*- coding: utf-8 -*-
import pygame
import sys
from lab02_ex2 import (
    inicializar_base_de_dados,
    motor_de_inferencia,
    iniciar_log_voo,
    fechar_log_voo
)

# =================================================================================
# 3. INTERFACE (Entradas/Saídas - I/O)
# =================================================================================
# Responsável pela visualização e interação com o utilizador.

def desenhar_sistema(tela, base_de_dados):
    """Desenha todos os elementos do sistema na tela."""
    tela.fill(base_de_dados['PRETO'])

    # Desenhar locais
    for posicao, coordenadas in base_de_dados['POSICOES'].items():
        cor = base_de_dados['VERDE'] if posicao == 'Base' else base_de_dados['AZUL']
        pygame.draw.circle(tela, cor, coordenadas, 20)
        fonte = pygame.font.Font(None, 36)
        texto = fonte.render(posicao, True, base_de_dados['BRANCO'])
        tela.blit(texto, (coordenadas[0] - 10, coordenadas[1] - 50))

    # Desenhar obstáculo
    pygame.draw.rect(tela, base_de_dados['AMARELO'], (
        base_de_dados['LARGURA'] // 2 - base_de_dados['OBSTACULO_LARGURA'] // 2,
        base_de_dados['obstaculo_y'],
        base_de_dados['OBSTACULO_LARGURA'],
        base_de_dados['OBSTACULO_ALTURA']
    ))

    # Desenhar rota alternativa
    if base_de_dados['evitando_obstaculo'] and base_de_dados['rota_alternativa']:
        pygame.draw.line(tela, base_de_dados['AMARELO'], base_de_dados['posicao_drone'], base_de_dados['rota_alternativa'], 2)
        pygame.draw.line(tela, base_de_dados['AMARELO'], base_de_dados['rota_alternativa'], base_de_dados['destino_original'], 2)

    # Desenhar drone
    pygame.draw.circle(tela, base_de_dados['VERMELHO'],
                      (int(base_de_dados['posicao_drone'][0]), int(base_de_dados['posicao_drone'][1])),
                      base_de_dados['RAIO_DRONE'])

    # Desenhar informações
    fonte_info = pygame.font.Font(None, 24)
    info = ("Drone: " + str(base_de_dados['nome_drone']) +
            " | Estado: " + str(base_de_dados['estado_drone']) +
            " | Posição: " + str(base_de_dados['posicao_atual']) +
            " | Próximo Destino: " + str(base_de_dados['proximo_destino']))
    texto_info = fonte_info.render(info, True, base_de_dados['BRANCO'])
    tela.blit(texto_info, (10, 10))

    # Desenhar alerta de colisão
    if base_de_dados['estado_drone'] == 'Colisão':
        fonte = pygame.font.Font(None, 72)
        texto = fonte.render("COLISÃO!", True, base_de_dados['VERMELHO'])
        tela.blit(texto, (base_de_dados['LARGURA'] // 2 - 120, base_de_dados['ALTURA'] // 2 - 30))

def simular_sistema():
    """
    Função principal que inicializa o sistema e corre o loop da simulação.
    """
    pygame.init()
    iniciar_log_voo()  # Inicia o logging do voo

    # Inicializa a base de dados
    base_de_dados = inicializar_base_de_dados()

    # Configura a tela (saída)
    tela = pygame.display.set_mode((base_de_dados['LARGURA'], base_de_dados['ALTURA']))
    pygame.display.set_caption("Simulação de Drone - Arquitetura Baseada em Regras")
    clock = pygame.time.Clock()

    # Loop principal da simulação
    try:
        while True:
            # Lidar com entradas (input)
            for evento in pygame.event.get():
                if evento.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Tomada de Decisão: aplicar regras
            base_de_dados = motor_de_inferencia(base_de_dados)

            # Saída: desenhar o estado atual
            desenhar_sistema(tela, base_de_dados)

            pygame.display.flip()
            clock.tick(30)
    finally:
        fechar_log_voo()  # Garante que o log é fechado ao sair

if __name__ == '__main__':
    simular_sistema()
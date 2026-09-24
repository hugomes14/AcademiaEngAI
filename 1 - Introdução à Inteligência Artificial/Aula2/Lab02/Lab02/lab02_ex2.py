# -*- coding: utf-8 -*-
import pygame
import math
import datetime

# Variável global para o ficheiro de log
log_file = None

def iniciar_log_voo():
    """Cria um ficheiro de log para o voo com base na data e hora de início."""
    global log_file
    now = datetime.datetime.now()
    # Formato do nome do ficheiro: log_voo_AAAA-MM-DD_HH-MM-SS.txt
    nome_ficheiro = now.strftime("log_voo_%Y-%m-%d_%H-%M-%S.txt")
    log_file = open(nome_ficheiro, 'w', encoding='utf-8')
    log_explicabilidade(
        regra="Inicialização",
        estado_anterior="Nenhum",
        estado_novo="Estado Seguro",
        justificacao="O sistema foi inicializado e o drone está pronto na base."
    )

def fechar_log_voo():
    """Fecha o ficheiro de log se estiver aberto."""
    global log_file
    if log_file:
        log_file.close()
        log_file = None

def log_explicabilidade(regra, estado_anterior, estado_novo, justificacao):
    """Regista uma decisão ou evento no ficheiro de log."""
    global log_file
    if log_file:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = (
            "[" + timestamp + "] - Regra Ativada: " + regra + "\n"
            "  - Estado Anterior: " + str(estado_anterior) + "\n"
            "  - Estado Novo: " + str(estado_novo) + "\n"
            "  - Justificação: " + justificacao + "\n"
            "--------------------------------------------------\n"
        )
        log_file.write(log_entry)

# =================================================================================
# 1. BASE DE CONHECIMENTO (Knowledge Base)
# =================================================================================

# ---------------------------------------------------------------------------------
# 1.1. Base de Dados (Database)
# ---------------------------------------------------------------------------------
# Contém todos os dados estáticos e o estado dinâmico do sistema.

def inicializar_base_de_dados():
    """
    Define e retorna o estado inicial do sistema (os factos).
    """
    # Dados Estáticos (Símbolos/Constantes)
    base_de_dados = {
        # Configurações da tela
        'LARGURA': 800,
        'ALTURA': 600,
        # Cores
        'PRETO': (0, 0, 0),
        'BRANCO': (255, 255, 255),
        'VERMELHO': (255, 0, 0),
        'VERDE': (0, 255, 0),
        'AZUL': (0, 0, 255),
        'AMARELO': (255, 255, 0),
        # Posições dos locais
        'POSICOES': {
            'Base': (400, 550),
            'A': (200, 200),
            'B': (600, 200),
            'C': (200, 400),
            'D': (600, 400)
        },
        # Configurações do drone e obstáculo
        'RAIO_DRONE': 10,
        'OBSTACULO_LARGURA': 20,
        'OBSTACULO_ALTURA': 200,
        'OBSTACULO_VELOCIDADE': 2
    }

    # Estado Dinâmico (Factos Iniciais)
    base_de_dados.update({
        'nome_drone': 'coruja',
        'estado_drone': 'Descolagem',
        'posicao_atual': 'Base',
        'posicao_drone': base_de_dados['POSICOES']['Base'],
        'destino': base_de_dados['POSICOES']['A'],
        'destino_original': None,
        'proximo_destino': 'A',
        'movimento_completo': False,
        'obstaculo_y': (base_de_dados['ALTURA'] - base_de_dados['OBSTACULO_ALTURA']) // 2,
        'obstaculo_direcao': 1,
        'rota_alternativa': None,
        'evitando_obstaculo': False
    })

    return base_de_dados

# ---------------------------------------------------------------------------------
# 1.2. Base de Regras (Rule Base)
# ---------------------------------------------------------------------------------
# Contém todas as regras e lógicas que o sistema usa para operar.

def regra_mover_obstaculo(base_de_dados):
    base_de_dados['obstaculo_y'] += (
        base_de_dados['OBSTACULO_VELOCIDADE'] * base_de_dados['obstaculo_direcao']
    )
    if (
        base_de_dados['obstaculo_y'] <= 0
        or base_de_dados['obstaculo_y'] + base_de_dados['OBSTACULO_ALTURA'] >= base_de_dados['ALTURA']
    ):
        base_de_dados['obstaculo_direcao'] *= -1

    return base_de_dados

def verificar_colisao(base_de_dados, posicao_drone):
    drone_x, drone_y = posicao_drone
    obstaculo_x = base_de_dados['LARGURA'] // 2 - base_de_dados['OBSTACULO_LARGURA'] // 2

    drone_rect = pygame.Rect(
        drone_x - base_de_dados['RAIO_DRONE'], drone_y - base_de_dados['RAIO_DRONE'],
        base_de_dados['RAIO_DRONE'] * 2, base_de_dados['RAIO_DRONE'] * 2
    )
    obstaculo_rect = pygame.Rect(
        obstaculo_x, base_de_dados['obstaculo_y'],
        base_de_dados['OBSTACULO_LARGURA'], base_de_dados['OBSTACULO_ALTURA']
    )
    return drone_rect.colliderect(obstaculo_rect)

def prever_colisao(base_de_dados):
    x1, y1 = base_de_dados['posicao_drone']
    x2, y2 = base_de_dados['destino']
    for i in range(10):
        t = i / 10.0
        x = x1 + (x2 - x1) * t
        y = y1 + (y2 - y1) * t
        if verificar_colisao(base_de_dados, (x, y)):
            return True
    return False

def calcular_rota_alternativa(base_de_dados):
    drone_x, drone_y = base_de_dados['posicao_drone']
    obstaculo_x = base_de_dados['LARGURA'] // 2
    destino_x, destino_y = base_de_dados['destino']

    if drone_x < obstaculo_x:
        ponto_desvio = (
            obstaculo_x - base_de_dados['OBSTACULO_LARGURA'] - 30,
            (drone_y + destino_y) / 2
        )
    else:
        ponto_desvio = (
            obstaculo_x + base_de_dados['OBSTACULO_LARGURA'] + 30,
            (drone_y + destino_y) / 2
        )
    return ponto_desvio

def regra_evitar_obstaculo(base_de_dados):
    if not base_de_dados['evitando_obstaculo'] and prever_colisao(base_de_dados):
        estado_anterior = base_de_dados['estado_drone']
        base_de_dados['evitando_obstaculo'] = True
        base_de_dados['rota_alternativa'] = calcular_rota_alternativa(base_de_dados)
        base_de_dados['destino_original'] = base_de_dados['destino']
        base_de_dados['destino'] = base_de_dados['rota_alternativa']
        log_explicabilidade(
            regra="Evitar Obstáculo",
            estado_anterior=estado_anterior,
            estado_novo="Evitando Obstáculo",
            justificacao="Colisão prevista. A calcular e seguir rota alternativa."
        )

    if base_de_dados['evitando_obstaculo'] and base_de_dados['movimento_completo']:
        estado_anterior = base_de_dados['estado_drone']
        base_de_dados['destino'] = base_de_dados['destino_original']
        base_de_dados['rota_alternativa'] = None
        base_de_dados['evitando_obstaculo'] = False
        base_de_dados['movimento_completo'] = False
        log_explicabilidade(
            regra="Retomar Rota Original",
            estado_anterior=estado_anterior,
            estado_novo="Em Rota",
            justificacao="Desvio concluído. A regressar ao destino original."
        )

    return base_de_dados

def regra_mover_drone(base_de_dados):
    if base_de_dados['estado_drone'] == 'Colisão':
        return base_de_dados

    velocidade = 5
    x1, y1 = base_de_dados['posicao_drone']
    x2, y2 = base_de_dados['destino']

    dx, dy = x2 - x1, y2 - y1
    distancia = math.sqrt(dx**2 + dy**2)

    if distancia >= velocidade:
        fator = velocidade / distancia
        novo_x = x1 + dx * fator
        novo_y = y1 + dy * fator
        base_de_dados['posicao_drone'] = (novo_x, novo_y)
    else:
        base_de_dados['posicao_drone'] = base_de_dados['destino']
        base_de_dados['movimento_completo'] = True

    return base_de_dados

def regra_atualizar_estado_navegacao(base_de_dados):
    if base_de_dados['movimento_completo'] and not base_de_dados['evitando_obstaculo']:
        if base_de_dados['estado_drone'] == 'Descolagem':
            base_de_dados['estado_drone'] = 'Em Rota'

        posicao_anterior = base_de_dados['posicao_atual']
        base_de_dados['posicao_atual'] = base_de_dados['proximo_destino']

        ordem = ['A', 'B', 'C', 'D', 'Base']
        idx = ordem.index(base_de_dados['posicao_atual'])
        proxima_posicao = ordem[(idx + 1) % len(ordem)]

        base_de_dados['proximo_destino'] = proxima_posicao
        base_de_dados['destino'] = base_de_dados['POSICOES'][proxima_posicao]
        base_de_dados['movimento_completo'] = False

        log_explicabilidade(
            regra="Atualizar Navegação",
            estado_anterior="Em trânsito para " + str(posicao_anterior),
            estado_novo="Chegou a " + str(base_de_dados['posicao_atual']) + ". Próximo destino: " + str(proxima_posicao),
            justificacao="O drone completou o trajeto e inicia o próximo segmento."
        )
    return base_de_dados

def regra_verificar_colisao_real(base_de_dados):
    if verificar_colisao(base_de_dados, base_de_dados['posicao_drone']):
        if base_de_dados['estado_drone'] != 'Colisão':
            estado_anterior = base_de_dados['estado_drone']
            base_de_dados['estado_drone'] = 'Colisão'
            log_explicabilidade(
                regra="Verificar Colisão Real",
                estado_anterior=estado_anterior,
                estado_novo="Colisão",
                justificacao="O drone colidiu com o obstáculo. Operação abortada."
            )
    return base_de_dados



# =================================================================================
# 2. TOMADA DE DECISÃO (Decision Making)
# =================================================================================
# O motor de inferência que aplica as regras à base de dados.

def motor_de_inferencia(base_de_dados):
    """
    Aplica a sequência de regras para atualizar o estado do sistema.
    """
    if base_de_dados['estado_drone'] != 'Colisão':
        base_de_dados = regra_evitar_obstaculo(base_de_dados)
        base_de_dados = regra_mover_drone(base_de_dados)
        base_de_dados = regra_atualizar_estado_navegacao(base_de_dados)
        base_de_dados = regra_mover_obstaculo(base_de_dados)
        base_de_dados = regra_verificar_colisao_real(base_de_dados)

    return base_de_dados

"""
Ambientes simples para o Lab 08.1 — Aprendizagem por Reforço.

Este ficheiro evita dependências pesadas e torna o laboratório reproduzível:
1. MultiArmedBandit: simula várias rotas de voo com recompensas incertas.
2. GridWorld: simula uma grelha 10x10 para treino com Q-Learning.

Os scripts principais usam estas classes para que o laboratório funcione
mesmo sem Gymnasium instalado.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass
class MultiArmedBandit:
    """Simulador de N rotas de voo, cada uma com recompensa média diferente."""

    route_names: List[str]
    means: np.ndarray
    stds: np.ndarray
    seed: int = 42

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.n_actions = len(self.route_names)
        self.optimal_action = int(np.argmax(self.means))
        self.optimal_mean = float(np.max(self.means))

    def step(self, action: int) -> float:
        """Executa uma ação e devolve uma recompensa contínua simulada."""
        return float(self.rng.normal(self.means[action], self.stds[action]))

    def describe(self) -> Dict[str, object]:
        """Devolve uma descrição simples do ambiente."""
        return {
            "num_rotas": self.n_actions,
            "rotas": self.route_names,
            "melhor_rota_real": self.route_names[self.optimal_action],
            "recompensa_media_otima_real": self.optimal_mean,
        }


class GridWorld:
    """
    Ambiente de navegação em grelha.

    Estados: posições (linha, coluna), convertidas para índice inteiro.
    Ações:
        0 = cima
        1 = baixo
        2 = esquerda
        3 = direita
    Recompensas:
        +10 ao chegar ao destino
        -10 ao colidir com obstáculo
        -0.1 por passo normal
    """

    ACTIONS = {
        0: (-1, 0),
        1: (1, 0),
        2: (0, -1),
        3: (0, 1),
    }

    ACTION_SYMBOLS = {
        0: "↑",
        1: "↓",
        2: "←",
        3: "→",
    }

    def __init__(
        self,
        size: int = 10,
        start: Tuple[int, int] = (0, 0),
        goal: Tuple[int, int] = (9, 9),
        obstacles: List[Tuple[int, int]] | None = None,
        seed: int = 42,
    ) -> None:
        self.size = size
        self.start = start
        self.goal = goal
        self.obstacles = set(
            obstacles
            if obstacles is not None
            else [
                (1, 3), (2, 3), (3, 3), (4, 3),
                (4, 4), (4, 5), (4, 6),
                (6, 1), (6, 2), (6, 3),
                (7, 6), (8, 6),
            ]
        )
        self.rng = np.random.default_rng(seed)
        self.n_states = self.size * self.size
        self.n_actions = 4
        self.position = self.start

    def reset(self) -> int:
        """Reinicia o episódio e devolve o estado inicial."""
        self.position = self.start
        return self.state_to_index(self.position)

    def state_to_index(self, position: Tuple[int, int]) -> int:
        """Converte posição (linha, coluna) para índice inteiro."""
        row, col = position
        return row * self.size + col

    def index_to_state(self, state_index: int) -> Tuple[int, int]:
        """Converte índice inteiro para posição (linha, coluna)."""
        row = state_index // self.size
        col = state_index % self.size
        return (row, col)

    def step(self, action: int) -> Tuple[int, float, bool, Dict[str, object]]:
        """Executa uma ação no ambiente."""
        row, col = self.position
        d_row, d_col = self.ACTIONS[action]
        next_row = int(np.clip(row + d_row, 0, self.size - 1))
        next_col = int(np.clip(col + d_col, 0, self.size - 1))
        next_position = (next_row, next_col)

        if next_position in self.obstacles:
            self.position = next_position
            return self.state_to_index(self.position), -10.0, True, {"motivo": "obstaculo"}

        if next_position == self.goal:
            self.position = next_position
            return self.state_to_index(self.position), 10.0, True, {"motivo": "destino"}

        self.position = next_position
        return self.state_to_index(self.position), -0.1, False, {"motivo": "passo"}

    def sample_action(self) -> int:
        """Escolhe uma ação aleatória."""
        return int(self.rng.integers(0, self.n_actions))

    def is_terminal_position(self, position: Tuple[int, int]) -> bool:
        """Indica se a posição termina o episódio."""
        return position == self.goal or position in self.obstacles

    def describe(self) -> Dict[str, object]:
        """Devolve uma descrição simples do ambiente."""
        return {
            "tipo": "GridWorld discreto",
            "tamanho": f"{self.size}x{self.size}",
            "num_estados": self.n_states,
            "num_acoes": self.n_actions,
            "inicio": self.start,
            "destino": self.goal,
            "num_obstaculos": len(self.obstacles),
            "recompensas": {
                "destino": 10.0,
                "obstaculo": -10.0,
                "passo": -0.1,
            },
        }


def moving_average(values: np.ndarray, window: int = 100) -> np.ndarray:
    """Calcula média móvel com janela fixa."""
    values = np.asarray(values, dtype=float)
    if len(values) < window:
        return np.array([])
    return np.convolve(values, np.ones(window) / window, mode="valid")

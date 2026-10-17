"""Um Transformer decoder pequeno, com atenção escrita passo a passo.

Formas: B = tamanho do lote; T = caracteres; C = dimensão do embedding;
H = cabeças; D = C/H. Não usa implementações prontas de atenção.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    context_size: int = 64
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 2
    dropout: float = 0.1

    def __post_init__(self):
        if self.vocab_size < 2:
            raise ValueError('O vocabulário precisa de pelo menos dois caracteres.')
        if min(self.context_size, self.d_model, self.n_heads, self.n_layers) < 1:
            raise ValueError('As dimensões e o número de camadas devem ser positivos.')
        if self.d_model % self.n_heads:
            raise ValueError('d_model tem de ser divisível por n_heads.')
        if not 0 <= self.dropout < 1:
            raise ValueError('dropout deve estar no intervalo [0, 1).')


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        # Três projeções aprendidas: o que procuro, o que ofereço e o conteúdo.
        self.query = nn.Linear(config.d_model, config.d_model, bias=False)
        self.key = nn.Linear(config.d_model, config.d_model, bias=False)
        self.value = nn.Linear(config.d_model, config.d_model, bias=False)
        self.projection = nn.Linear(config.d_model, config.d_model)
        self.dropout = nn.Dropout(config.dropout)
        # True identifica posições futuras, às quais é proibido atender.
        self.register_buffer('future_mask', torch.triu(
            torch.ones(config.context_size, config.context_size, dtype=torch.bool), 1))

    def forward(self, x, capture_attention=False):
        batch, time, channels = x.shape

        def separate_heads(projected):
            return projected.view(batch, time, self.n_heads,
                                  self.head_dim).transpose(1, 2)

        q = separate_heads(self.query(x))  # (B, H, T, D)
        k = separate_heads(self.key(x))
        v = separate_heads(self.value(x))

        # Cada query compara-se com todas as keys. A escala controla magnitudes.
        scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        scores = scores.masked_fill(self.future_mask[:time, :time], float('-inf'))
        weights = F.softmax(scores, dim=-1)  # (B, H, T, T), soma por linha = 1
        # A mistura ponderada das values produz o contexto para cada posição.
        context = self.dropout(weights) @ v
        context = context.transpose(1, 2).contiguous().view(batch, time, channels)
        output = self.dropout(self.projection(context))
        # Capturamos pesos ANTES do dropout para serem probabilidades legíveis.
        return output, weights.detach() if capture_attention else None


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.d_model)
        self.attention = CausalSelfAttention(config)
        self.norm_mlp = nn.LayerNorm(config.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(config.d_model, 4 * config.d_model),
            nn.GELU(),
            nn.Linear(4 * config.d_model, config.d_model),
            nn.Dropout(config.dropout),
        )

    def forward(self, x, capture_attention=False):
        # Pre-norm: normalizar antes de transformar; somar preserva informação.
        attended, weights = self.attention(self.norm_attention(x), capture_attention)
        x = x + attended
        x = x + self.mlp(self.norm_mlp(x))
        return x, weights


class MiniTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.context_size, config.d_model)
        self.blocks = nn.ModuleList(TransformerBlock(config)
                                    for _ in range(config.n_layers))
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, tokens, capture_attention=False):
        if tokens.ndim != 2 or not 0 < tokens.shape[1] <= self.config.context_size:
            raise ValueError('Esperado um lote (B, T) com 1 <= T <= context_size.')
        positions = torch.arange(tokens.shape[1], device=tokens.device)
        x = self.token_embedding(tokens) + self.position_embedding(positions)
        attention_maps = []
        for block in self.blocks:
            x, attention = block(x, capture_attention)
            if capture_attention:
                attention_maps.append(attention)
        return self.lm_head(self.final_norm(x)), attention_maps

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens=120, temperature=0.8, generator=None):
        if temperature <= 0 or max_new_tokens < 0:
            raise ValueError('temperature deve ser positiva e max_new_tokens >= 0.')
        if tokens.ndim != 2 or tokens.shape[1] == 0:
            raise ValueError('A geração precisa de pelo menos um token de contexto.')
        was_training = self.training
        self.eval()
        try:
            for _ in range(max_new_tokens):
                # Apenas a janela entra no modelo; o texto completo é preservado.
                logits, _ = self(tokens[:, -self.config.context_size:])
                probabilities = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, 1, generator=generator)
                tokens = torch.cat((tokens, next_token), dim=1)
        finally:
            self.train(was_training)
        return tokens

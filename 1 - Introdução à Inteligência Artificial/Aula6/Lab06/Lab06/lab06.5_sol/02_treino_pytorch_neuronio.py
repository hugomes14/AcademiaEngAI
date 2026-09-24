# -*- coding: utf-8 -*-
"""
Lab 06.5 - Treino de uma rede neural com 1 neurónio linear em PyTorch.

O modelo tem a forma:
    y = w1*x1 + w2*x2 + ... + wn*xn + b

Sem função de ativação não linear, esta arquitetura é equivalente a uma regressão linear múltipla.
"""

from pathlib import Path
import pickle
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

RAIZ = Path(__file__).resolve().parent
DIR_PREP = RAIZ / "artefactos" / "preprocessamento"
DIR_MODELOS = RAIZ / "artefactos" / "modelos"
DIR_PREV = RAIZ / "artefactos" / "previsoes"
DIR_METRICAS = RAIZ / "artefactos" / "metricas"

DIR_MODELOS.mkdir(parents=True, exist_ok=True)
DIR_PREV.mkdir(parents=True, exist_ok=True)
DIR_METRICAS.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 06.5 - TREINO PYTORCH: REDE NEURAL COM 1 NEURÓNIO")
print("=" * 80)

with open(DIR_PREP / "dados_processados.pkl", "rb") as f:
    dados = pickle.load(f)

X_train = torch.tensor(dados["X_train_proc"], dtype=torch.float32)
X_val = torch.tensor(dados["X_val_proc"], dtype=torch.float32)
X_test = torch.tensor(dados["X_test_proc"], dtype=torch.float32)

# Para tornar a otimização com Adam mais estável, a variável alvo também é
# padronizada com média/desvio calculados APENAS no treino.
# As previsões e os coeficientes são depois convertidos de volta para minutos.
y_train_original = dados["y_train_array"].astype(np.float32)
y_val_original = dados["y_val_array"].astype(np.float32)
y_test_original = dados["y_test_array"].astype(np.float32)
y_media = float(y_train_original.mean())
y_desvio = float(y_train_original.std())

y_train = torch.tensor((y_train_original - y_media) / y_desvio, dtype=torch.float32).view(-1, 1)
y_val = torch.tensor((y_val_original - y_media) / y_desvio, dtype=torch.float32).view(-1, 1)
y_test = torch.tensor((y_test_original - y_media) / y_desvio, dtype=torch.float32).view(-1, 1)
feature_names = dados["feature_names"]

print(f"Features de entrada: {X_train.shape[1]}")
print(f"Exemplos de treino: {X_train.shape[0]}")

# -----------------------------------------------------------------------------
# Classe do modelo: 1 camada linear, 1 saída, sem ativação.
# -----------------------------------------------------------------------------
class RegressaoNeuronio(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.linear(x)

# Reprodutibilidade.
torch.manual_seed(42)
np.random.seed(42)

modelo = RegressaoNeuronio(n_features=X_train.shape[1])
criterio = nn.MSELoss()
# Adam acelera a convergência face ao SGD puro neste exemplo didático.
otimizador = torch.optim.Adam(modelo.parameters(), lr=0.05)

epocas_max = 5000
patience = 700
min_delta = 1e-7
melhor_val_loss = float("inf")
melhor_estado = None
sem_melhoria = 0
historico = []

inicio = time.time()

for epoca in range(1, epocas_max + 1):
    # Modo de treino: ativa comportamentos próprios de treino se existirem.
    modelo.train()

    # 1) Forward pass: calcula previsões.
    y_pred_train = modelo(X_train)

    # 2) Loss: mede o erro médio quadrático.
    loss_train = criterio(y_pred_train, y_train)

    # 3) Backward pass: calcula gradientes.
    otimizador.zero_grad()
    loss_train.backward()

    # 4) Atualização dos pesos.
    otimizador.step()

    # Validação sem cálculo de gradientes.
    modelo.eval()
    with torch.no_grad():
        y_pred_val = modelo(X_val)
        loss_val = criterio(y_pred_val, y_val)

    historico.append(
        {
            "epoca": epoca,
            "loss_treino": float(loss_train.item()),
            "loss_validacao": float(loss_val.item()),
        }
    )

    if loss_val.item() < melhor_val_loss - min_delta:
        melhor_val_loss = loss_val.item()
        melhor_estado = {k: v.detach().clone() for k, v in modelo.state_dict().items()}
        sem_melhoria = 0
    else:
        sem_melhoria += 1

    if epoca == 1 or epoca % 500 == 0:
        print(
            f"Época {epoca:04d} | "
            f"loss treino={loss_train.item():.4f} | "
            f"loss validação={loss_val.item():.4f}"
        )

    if sem_melhoria >= patience:
        print(f"\nEarly stopping ativado na época {epoca}.")
        break

if melhor_estado is not None:
    modelo.load_state_dict(melhor_estado)

tempo = time.time() - inicio
print(f"\nTempo de treino PyTorch: {tempo:.2f} segundos")
print(f"Melhor loss de validação na escala padronizada do alvo: {melhor_val_loss:.6f}")

modelo.eval()
with torch.no_grad():
    # As previsões saem na escala padronizada do alvo.
    # Convertem-se de volta para minutos para avaliação e interpretação.
    pred_train = modelo(X_train).numpy().ravel() * y_desvio + y_media
    pred_val = modelo(X_val).numpy().ravel() * y_desvio + y_media
    pred_test = modelo(X_test).numpy().ravel() * y_desvio + y_media

# Guardar modelo e metadados.
torch.save(
    {
        "model_state_dict": modelo.state_dict(),
        "n_features": X_train.shape[1],
        "feature_names": feature_names,
        "melhor_val_loss": melhor_val_loss,
        "tempo_treino_segundos": tempo,
        "target_media_treino": y_media,
        "target_desvio_treino": y_desvio,
    },
    DIR_MODELOS / "modelo_pytorch_1_neuronio.pt",
)

# Guardar previsões.
pd.DataFrame({"y_real": dados["y_test_array"], "y_pred_pytorch": pred_test}).to_csv(
    DIR_PREV / "previsoes_pytorch_teste.csv", index=False
)
pd.DataFrame({"y_real": dados["y_val_array"], "y_pred_pytorch": pred_val}).to_csv(
    DIR_PREV / "previsoes_pytorch_validacao.csv", index=False
)

# Guardar histórico de treino.
historico_df = pd.DataFrame(historico)
historico_df.to_csv(DIR_METRICAS / "historico_treino_pytorch.csv", index=False)

# Guardar pesos convertidos para a escala original do alvo.
# Se y_scaled = w*x + b, então y_minutos = y_media + y_desvio*(w*x + b).
pesos = modelo.linear.weight.detach().numpy().ravel() * y_desvio
bias = y_media + y_desvio * float(modelo.linear.bias.detach().numpy().ravel()[0])
coef_df = pd.DataFrame({"feature": feature_names, "peso_pytorch": pesos})
coef_df.loc[len(coef_df)] = ["bias/intercept", bias]
coef_df.to_csv(DIR_METRICAS / "coeficientes_pytorch.csv", index=False)

print("\nPesos do neurónio PyTorch:")
print(coef_df)
print("\nModelo, previsões, histórico e coeficientes PyTorch guardados com sucesso.")

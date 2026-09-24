# Lab 06.3 — Séries Temporais
# Script 02 — Pré-processamento e Engenharia de Features
# Objetivo: criar variáveis temporais, lags, médias móveis, split temporal e escalonamento.
#
# Nota crítica:
# Em séries temporais, o conjunto de teste deve representar o futuro.
# Por isso, NÃO se usa split aleatório. O teste fica com os dados mais recentes.
# Esta decisão evita data leakage, ou seja, evita que informação futura influencie o treino.

from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib

from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parents[1]
DADOS_DIR = BASE_DIR / "dados"
ARTEFACTOS_DIR = BASE_DIR / "artefactos"
OBJ_DIR = ARTEFACTOS_DIR / "objetos"

OBJ_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 90)
print("SCRIPT 02 — PRÉ-PROCESSAMENTO E ENGENHARIA DE FEATURES")
print("=" * 90)

# -------------------------------------------------------------------------
# Leitura e preparação base
# -------------------------------------------------------------------------
voltagem = pd.read_csv(DADOS_DIR / "voltagem_bateria.csv")
missoes = pd.read_csv(DADOS_DIR / "missoes_diarias.csv")

voltagem["timestamp"] = pd.to_datetime(voltagem["timestamp"], errors="coerce")
missoes["data"] = pd.to_datetime(missoes["data"], errors="coerce")

voltagem = voltagem.dropna(subset=["timestamp", "voltagem"]).sort_values("timestamp")
missoes = missoes.dropna(subset=["data", "num_missoes"]).sort_values("data")

voltagem = voltagem.set_index("timestamp")
missoes = missoes.set_index("data")

voltagem = voltagem[~voltagem.index.duplicated(keep="first")]
missoes = missoes[~missoes.index.duplicated(keep="first")]

# -------------------------------------------------------------------------
# Garantir uma grelha temporal completa.
# Se existirem lacunas, a série é reindexada.
# Para valores em falta, usa-se interpolação temporal.
# -------------------------------------------------------------------------
voltagem = voltagem.reindex(pd.date_range(voltagem.index.min(), voltagem.index.max(), freq="min"))
voltagem.index.name = "timestamp"
voltagem["voltagem"] = voltagem["voltagem"].interpolate(method="time").ffill().bfill()

missoes = missoes.reindex(pd.date_range(missoes.index.min(), missoes.index.max(), freq="D"))
missoes.index.name = "data"
missoes["num_missoes"] = missoes["num_missoes"].interpolate(method="time").round().ffill().bfill()

print(f"[INFO] Série de voltagem após reindexação: {voltagem.shape}")
print(f"[INFO] Série de missões após reindexação: {missoes.shape}")

# -------------------------------------------------------------------------
# Features de calendário e lags — Voltagem
# -------------------------------------------------------------------------
df_v = voltagem.copy()
df_v["alvo"] = df_v["voltagem"]

df_v["minuto"] = df_v.index.minute
df_v["hora"] = df_v.index.hour
df_v["dia_semana"] = df_v.index.dayofweek
df_v["dia_mes"] = df_v.index.day
df_v["mes"] = df_v.index.month
df_v["fim_semana"] = (df_v.index.dayofweek >= 5).astype(int)

# Codificação cíclica para variáveis periódicas.
# Ajuda modelos lineares a perceber que 23h e 0h estão próximas.
df_v["hora_sin"] = np.sin(2 * np.pi * df_v["hora"] / 24)
df_v["hora_cos"] = np.cos(2 * np.pi * df_v["hora"] / 24)
df_v["minuto_sin"] = np.sin(2 * np.pi * df_v["minuto"] / 60)
df_v["minuto_cos"] = np.cos(2 * np.pi * df_v["minuto"] / 60)
df_v["dia_semana_sin"] = np.sin(2 * np.pi * df_v["dia_semana"] / 7)
df_v["dia_semana_cos"] = np.cos(2 * np.pi * df_v["dia_semana"] / 7)

# Lags adaptados a uma série minuto a minuto.
for lag in [1, 5, 15, 30, 60, 120, 1440]:
    df_v[f"voltagem_lag_{lag}"] = df_v["alvo"].shift(lag)

# Janelas móveis. O shift(1) garante que a observação atual não entra no cálculo.
for janela in [15, 60, 180]:
    df_v[f"voltagem_media_movel_{janela}"] = df_v["alvo"].shift(1).rolling(janela).mean()
    df_v[f"voltagem_std_movel_{janela}"] = df_v["alvo"].shift(1).rolling(janela).std()

df_v = df_v.dropna()

# -------------------------------------------------------------------------
# Features de calendário e lags — Missões
# -------------------------------------------------------------------------
df_m = missoes.copy()
df_m["alvo"] = df_m["num_missoes"]

df_m["dia_semana"] = df_m.index.dayofweek
df_m["dia_mes"] = df_m.index.day
df_m["mes"] = df_m.index.month
df_m["trimestre"] = df_m.index.quarter
df_m["dia_ano"] = df_m.index.dayofyear
df_m["semana_ano"] = df_m.index.isocalendar().week.astype(int)
df_m["fim_semana"] = (df_m.index.dayofweek >= 5).astype(int)

df_m["dia_semana_sin"] = np.sin(2 * np.pi * df_m["dia_semana"] / 7)
df_m["dia_semana_cos"] = np.cos(2 * np.pi * df_m["dia_semana"] / 7)
df_m["mes_sin"] = np.sin(2 * np.pi * df_m["mes"] / 12)
df_m["mes_cos"] = np.cos(2 * np.pi * df_m["mes"] / 12)
df_m["dia_ano_sin"] = np.sin(2 * np.pi * df_m["dia_ano"] / 365)
df_m["dia_ano_cos"] = np.cos(2 * np.pi * df_m["dia_ano"] / 365)

# Lags adaptados a uma série diária.
for lag in [1, 2, 3, 7, 14, 28]:
    df_m[f"num_missoes_lag_{lag}"] = df_m["alvo"].shift(lag)

for janela in [7, 14, 28]:
    df_m[f"num_missoes_media_movel_{janela}"] = df_m["alvo"].shift(1).rolling(janela).mean()
    df_m[f"num_missoes_std_movel_{janela}"] = df_m["alvo"].shift(1).rolling(janela).std()

df_m = df_m.dropna()

# -------------------------------------------------------------------------
# Split temporal e escalonamento
# -------------------------------------------------------------------------
dados_processados = {}

for nome, df, target_original in [
    ("voltagem", df_v, "voltagem"),
    ("missoes", df_m, "num_missoes"),
]:
    print("\n" + "-" * 90)
    print(f"[SPLIT TEMPORAL] {nome.upper()}")
    print("-" * 90)

    # Não se usa a coluna alvo original nem a coluna "alvo" como feature.
    features = [c for c in df.columns if c not in [target_original, "alvo"]]
    X = df[features].copy()
    y = df["alvo"].copy()

    indice_split = int(len(df) * 0.8)

    X_train = X.iloc[:indice_split].copy()
    X_test = X.iloc[indice_split:].copy()
    y_train = y.iloc[:indice_split].copy()
    y_test = y.iloc[indice_split:].copy()

    print(f"[INFO] Total após features: {len(df)}")
    print(f"[INFO] Treino: {len(X_train)} observações")
    print(f"[INFO] Teste: {len(X_test)} observações")
    print(f"[INFO] Período treino: {X_train.index.min()} a {X_train.index.max()}")
    print(f"[INFO] Período teste: {X_test.index.min()} a {X_test.index.max()}")

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=features,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=features,
    )

    # Guardar tudo em pickle para os scripts seguintes.
    pacote = {
        "nome": nome,
        "features": features,
        "target": "alvo",
        "target_original": target_original,
        "df_features": df,
        "X_train": X_train,
        "X_test": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "scaler": scaler,
        "split_info": {
            "metodo": "split temporal 80/20",
            "tamanho_treino": int(len(X_train)),
            "tamanho_teste": int(len(X_test)),
            "inicio_treino": str(X_train.index.min()),
            "fim_treino": str(X_train.index.max()),
            "inicio_teste": str(X_test.index.min()),
            "fim_teste": str(X_test.index.max()),
        },
    }

    joblib.dump(pacote, OBJ_DIR / f"{nome}_dados_processados.pkl")
    df.to_csv(OBJ_DIR / f"{nome}_features.csv")

    dados_processados[nome] = pacote
    print(f"[OK] Objetos guardados: {OBJ_DIR / f'{nome}_dados_processados.pkl'}")

print("\n" + "=" * 90)
print("[OK] Pré-processamento concluído.")
print("=" * 90)

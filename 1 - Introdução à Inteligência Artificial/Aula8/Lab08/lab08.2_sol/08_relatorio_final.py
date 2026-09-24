# 08_relatorio_final.py
# Lab 08.2 — Algoritmos Genéticos
# Etapa 8: gerar relatório final em Markdown.

from pathlib import Path
import json
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
EDA_DIR = BASE_DIR / "artefactos" / "eda"
PREP_DIR = BASE_DIR / "artefactos" / "preprocessamento"
GA_DIR = BASE_DIR / "artefactos" / "ga_features"
AVAL_DIR = BASE_DIR / "artefactos" / "avaliacao"
HP_DIR = BASE_DIR / "artefactos" / "ga_hiperparametros"
ROTA_DIR = BASE_DIR / "artefactos" / "nsga2_rotas"
REL_DIR = BASE_DIR / "artefactos" / "relatorio"
REL_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ETAPA 8 — Relatório final")
print("=" * 80)

def ler_texto(path, fallback="Conteúdo não disponível."):
    path = Path(path)
    if path.exists():
        return path.read_text(encoding="utf-8")
    return fallback

def ler_tabela_md(path, fallback="Tabela não disponível."):
    path = Path(path)
    if path.exists():
        try:
            df = pd.read_csv(path)
            return df.to_markdown(index=False)
        except Exception:
            return fallback
    return fallback

metricas_md = ler_tabela_md(AVAL_DIR / "metricas_modelos.csv")
metricas_rf_md = ler_tabela_md(HP_DIR / "metricas_random_forest_ga.csv")
pareto_md = ler_tabela_md(ROTA_DIR / "frente_pareto_rotas.csv")

best_features_path = AVAL_DIR / "best_features.json"
if best_features_path.exists():
    best_features = json.loads(best_features_path.read_text(encoding="utf-8"))
    features_list = "\n".join(f"- `{feat}`" for feat in best_features.get("selected_features", []))
else:
    features_list = "- Não disponível."

report = f"""# RELATÓRIO FINAL — Lab 08.2: Algoritmos Genéticos

## 1. Introdução

Este laboratório demonstra o uso de Algoritmos Genéticos em dois tipos de problemas:

1. **Seleção de features** para classificação binária.
2. **Otimização de hiperparâmetros** de um modelo de Machine Learning.
3. **Otimização multiobjetivo de rotas**, com análise da Frente de Pareto.

A ideia central é trabalhar com uma população de soluções candidatas. Cada solução é avaliada por uma função de fitness e a população evolui através de seleção, cruzamento e mutação.

## 2. Dados e EDA

O projeto é autocontido. Se não existir um dataset real, o script inicial gera um dataset sintético de voos com features de sensores e uma variável-alvo binária: `incidente_reportado`.

Artefactos principais:

- `artefactos/eda/distribuicao_target.png`
- `artefactos/eda/histogramas_features.png`
- `artefactos/eda/heatmap_correlacoes.png`

![Distribuição do target](../eda/distribuicao_target.png)

![Heatmap de correlações](../eda/heatmap_correlacoes.png)

## 3. Pipeline de Pré-processamento

O pipeline separa treino e teste com estratificação. Depois aplica `StandardScaler`, ajustado apenas no treino, para evitar *data leakage*.

A estratificação preserva a proporção de classes no treino e no teste.

## 4. Algoritmo Genético para Seleção de Features

### Representação do indivíduo

Cada indivíduo é uma lista binária:

- `1`: a feature entra no modelo.
- `0`: a feature fica fora do modelo.

### Função de fitness

A fitness usa o F1-score médio em validação cruzada, com uma pequena penalização por complexidade. Isto incentiva modelos mais simples, sem perder desempenho.

### Features selecionadas

{features_list}

### Convergência

![Convergência do GA](../ga_features/convergencia_ga.png)

![Tamanho dos subconjuntos](../ga_features/tamanho_subconjuntos_ga.png)

## 5. Avaliação do Melhor Subconjunto

{metricas_md}

![Matriz de Confusão](../avaliacao/matriz_confusao_ga.png)

### Interpretação

O modelo com features selecionadas deve ser comparado com o modelo base que usa todas as features. Mesmo que o desempenho fique ligeiramente abaixo, o subconjunto pode ser útil por reduzir complexidade, custo de recolha de dados e risco de sobreajuste.

## 6. Otimização de Hiperparâmetros

Foi implementado um AG simples para procurar hiperparâmetros de um `RandomForestClassifier`.

{metricas_rf_md}

![Convergência dos hiperparâmetros](../ga_hiperparametros/convergencia_ga_hiperparametros.png)

## 7. Otimização Multiobjetivo de Rotas

Nesta etapa, cada indivíduo representa uma rota com waypoints intermédios. O algoritmo procura minimizar dois objetivos:

- tempo de voo;
- consumo de combustível.

O resultado é uma Frente de Pareto, ou seja, um conjunto de soluções não dominadas.

{pareto_md}

![Frente de Pareto](../nsga2_rotas/frente_pareto_rotas.png)

## 8. Conclusões

Os Algoritmos Genéticos são úteis quando o espaço de procura é grande, discreto, irregular ou difícil de otimizar com métodos tradicionais.

Principais aprendizagens:

- A função de fitness define o comportamento do algoritmo.
- A mutação ajuda a manter diversidade.
- O elitismo preserva boas soluções.
- A validação cruzada reduz o risco de escolher features que apenas funcionam por acaso no treino.
- A Frente de Pareto é adequada quando existem objetivos concorrentes.

## 9. Próximos passos

- Testar uma implementação com `deap`.
- Aumentar o número de gerações e a população.
- Comparar a seleção por GA com métodos como `SelectKBest`, Lasso ou Random Forest importance.
- Acrescentar custo de sensores à função de fitness.
- Usar dados reais de telemetria e manutenção.
"""

(REL_DIR / "RELATORIO_FINAL.md").write_text(report, encoding="utf-8")
(BASE_DIR / "RELATORIO_FINAL.md").write_text(report, encoding="utf-8")

print("Relatório criado em:")
print("-", REL_DIR / "RELATORIO_FINAL.md")
print("-", BASE_DIR / "RELATORIO_FINAL.md")
print("Etapa 8 concluída com sucesso.")

# -*- coding: utf-8 -*-
"""
Lab 06.5 - Geração automática do relatório final em Markdown.
"""

from pathlib import Path
import pandas as pd

RAIZ = Path(__file__).resolve().parent
DIR_METRICAS = RAIZ / "artefactos" / "metricas"
DIR_REL = RAIZ

print("=" * 80)
print("LAB 06.5 - RELATÓRIO FINAL")
print("=" * 80)

metricas = pd.read_csv(DIR_METRICAS / "metricas_modelos.csv")
coef = pd.read_csv(DIR_METRICAS / "comparacao_coeficientes.csv")
historico = pd.read_csv(DIR_METRICAS / "historico_treino_pytorch.csv")

try:
    tabela_metricas = metricas.to_markdown(index=False, floatfmt=".4f")
except Exception:
    tabela_metricas = metricas.to_string(index=False)

try:
    tabela_coef = coef.to_markdown(index=False, floatfmt=".6f")
except Exception:
    tabela_coef = coef.to_string(index=False)

melhor = metricas.sort_values("RMSE").iloc[0]
ultima_epoca = int(historico["epoca"].max())
melhor_loss_val = historico["loss_validacao"].min()

relatorio = f"""# RELATÓRIO FINAL — Lab 06.5: Regressão com Rede Neural de 1 Neurónio

## 1. Introdução

Este laboratório demonstra, de forma prática, que uma rede neural com apenas um neurónio linear é matematicamente equivalente a uma regressão linear múltipla.

A operação realizada pelo neurónio é:

```text
y = w1*x1 + w2*x2 + ... + wn*xn + b
```

Como não foi usada função de ativação não linear, o modelo PyTorch mantém uma relação linear entre as features e a duração prevista do voo.

## 2. Dados e preparação

O dataset usado foi `dados/voos_telemetria.csv`, com as seguintes variáveis:

- `distancia_planeada`
- `carga_util_kg`
- `altitude_media_m`
- `condicao_meteo`
- `duracao_voo_min`, a variável alvo

O pré-processamento aplicou:

- limpeza simples de duplicados e valores em falta;
- one-hot encoding à variável `condicao_meteo`;
- padronização das features com `StandardScaler`;
- divisão dos dados em treino, validação e teste: 70% / 15% / 15%.

O escalonamento foi ajustado apenas no conjunto de treino para evitar *data leakage*.

## 3. Modelo PyTorch

Foi treinado um modelo com uma única camada `nn.Linear(n_features, 1)`, sem função de ativação.

Configuração principal:

- função de perda: MSELoss;
- otimizador: Adam;
- early stopping com base na loss de validação;
- número de épocas executadas: {ultima_epoca};
- melhor loss de validação: {melhor_loss_val:.4f}.

![Curva de aprendizagem](imagens/05_curva_aprendizagem_pytorch.png)

## 4. Comparação com sklearn

Foi treinada também uma regressão linear múltipla com `LinearRegression`, usando exatamente os mesmos dados processados.

### Métricas no conjunto de teste

{tabela_metricas}

Melhor modelo pelo critério RMSE: **{melhor['modelo']}**.

## 5. Valores reais vs. previstos

### PyTorch

![Valores reais vs. previstos PyTorch](imagens/05_reais_vs_previstos_pytorch.png)

### sklearn

![Valores reais vs. previstos sklearn](imagens/05_reais_vs_previstos_sklearn.png)

## 6. Comparação de coeficientes

A tabela seguinte compara os pesos aprendidos pelo modelo PyTorch e pela regressão linear do sklearn no espaço de features já processadas/escaladas.

{tabela_coef}

![Comparação de coeficientes](imagens/05_comparacao_coeficientes.png)

## 7. Análise de resíduos

Os resíduos permitem perceber onde o modelo erra mais. Um modelo linear bem ajustado tende a apresentar resíduos centrados em zero e sem padrão claro face aos valores previstos.

![Distribuição dos resíduos](imagens/05_distribuicao_residuos_pytorch.png)

![Resíduos vs. previstos](imagens/05_residuos_vs_previstos_pytorch.png)

## 8. Conclusões

Este laboratório confirma a equivalência conceptual entre uma regressão linear múltipla e uma rede neural com um único neurónio linear.

As diferenças entre PyTorch e sklearn podem surgir porque:

- o sklearn resolve a regressão linear de forma fechada/analítica;
- o PyTorch aprende os pesos por otimização iterativa;
- a inicialização dos pesos, o learning rate, o número de épocas e o early stopping influenciam o resultado final.

Em problemas lineares simples, o sklearn é normalmente mais rápido, direto e interpretável. O PyTorch torna-se mais interessante quando se pretende evoluir para redes neurais com várias camadas, ativações não lineares, treino em GPU ou arquiteturas mais complexas.

## 9. Recomendações

- Experimentar `SGD` em vez de `Adam` e comparar a velocidade de convergência.
- Testar diferentes learning rates, por exemplo `0.001`, `0.01` e `0.1`.
- Adicionar uma camada escondida e uma ativação não linear para observar a passagem de modelo linear para rede neural não linear.
- Comparar o treino com e sem escalonamento para perceber o impacto na estabilidade dos gradientes.
"""

saida = DIR_REL / "RELATORIO_FINAL.md"
saida.write_text(relatorio, encoding="utf-8")
print(f"Relatório final gerado em: {saida}")

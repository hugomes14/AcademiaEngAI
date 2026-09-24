# RELATÓRIO FINAL — Lab 06.5: Regressão com Rede Neural de 1 Neurónio

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
- número de épocas executadas: 801;
- melhor loss de validação: 0.0039.

![Curva de aprendizagem](imagens/05_curva_aprendizagem_pytorch.png)

## 4. Comparação com sklearn

Foi treinada também uma regressão linear múltipla com `LinearRegression`, usando exatamente os mesmos dados processados.

### Métricas no conjunto de teste

| modelo                      |    MAE |      MSE |    RMSE |     R2 |
|:----------------------------|-------:|---------:|--------:|-------:|
| PyTorch - 1 neurónio linear | 7.7960 | 101.2773 | 10.0637 | 0.9967 |
| sklearn - Regressão Linear  | 7.8816 | 102.3564 | 10.1171 | 0.9966 |

Melhor modelo pelo critério RMSE: **PyTorch - 1 neurónio linear**.

## 5. Valores reais vs. previstos

### PyTorch

![Valores reais vs. previstos PyTorch](imagens/05_reais_vs_previstos_pytorch.png)

### sklearn

![Valores reais vs. previstos sklearn](imagens/05_reais_vs_previstos_sklearn.png)

## 6. Comparação de coeficientes

A tabela seguinte compara os pesos aprendidos pelo modelo PyTorch e pela regressão linear do sklearn no espaço de features já processadas/escaladas.

| feature                 |   peso_pytorch |   peso_sklearn |   diferenca_absoluta |
|:------------------------|---------------:|---------------:|---------------------:|
| altitude_media_m        |      -1.720515 |      -1.590820 |             0.129696 |
| bias/intercept          |     394.671057 |     394.796821 |             0.125764 |
| carga_util_kg           |       1.922863 |       2.046845 |             0.123983 |
| condicao_meteo_Bom      |     -41.776630 |     -41.570976 |             0.205655 |
| condicao_meteo_Moderado |     -26.680393 |     -27.206986 |             0.526592 |
| distancia_planeada      |     168.024353 |     167.696950 |             0.327403 |

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

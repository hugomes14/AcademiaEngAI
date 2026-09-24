# Lab 06.5 - Aprendizagem Profunda (Deep Learning) — Regressão com Rede Neural (1 Neurónio) — Estimar a Duração de Voo

**Tema:** Redes Neurais com PyTorch - Equivalência com Regressão Linear

## 1. Objetivo

O objetivo deste exercício é demonstrar que **uma rede neural com apenas 1 neurónio é matematicamente equivalente à Regressão Linear Múltipla**. Vamos construir este neurónio usando **PyTorch** e compará-lo com os modelos clássicos do Lab 5.1.

### Por que este exercício é importante?
- Compreender que redes neurais são uma **generalização** de modelos lineares
- Aprender os fundamentos de PyTorch: tensors, autograd, e loops de treino
- Estabelecer a base para redes neurais mais complexas (próximos labs)

## 2. Dados

Continuamos a trabalhar com o mesmo dataset `voos_telemetria.csv`:
- `distancia_planeada` (km)
- `carga_util_kg` (kg)
- `altitude_media_m` (metros)
- `condicao_meteo` (categórica: `Bom`, `Moderado`, `Adverso`)
- `duracao_voo_min` (minutos) — **variável alvo**

## 3. Tarefas

### 3.1. Preparação dos Dados
1. **Pré-processamento:**
   - Carrega e limpa os dados (igual ao Lab 5.1)
   - Aplica *one-hot encoding* à variável `condicao_meteo`
   - **Normaliza/Padroniza** as features (crucial para redes neurais!)
   - Converte os dados para **Tensors do PyTorch**

2. **Divisão:**
   - Separa em conjuntos de treino (70%), validação (15%) e teste (15%)

### 3.2. Construção do Modelo Neural

**Arquitetura: 1 Neurónio Linear**

Implementa um modelo com:
- **Input:** n features (após one-hot encoding)
- **Output:** 1 valor (duração prevista)
- **Função de ativação:** Nenhuma (identidade) — mantém linearidade
- **Função de perda:** MSE (Mean Squared Error)
- **Otimizador:** SGD ou Adam

```python
import torch
import torch.nn as nn

class RegressaoNeuronio(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.linear = nn.Linear(n_features, 1)
    
    def forward(self, x):
        return self.linear(x)
```

### 3.3. Treino do Modelo

Implementa o **loop de treino** completo:
1. Forward pass
2. Cálculo da loss
3. Backward pass (gradientes)
4. Atualização dos pesos

**Pontos de atenção:**
- Usa `model.train()` durante o treino
- Usa `model.eval()` e `torch.no_grad()` durante a validação
- Monitoriza a loss de treino e validação por época
- Implementa **early stopping** (opcional)

### 3.4. Comparação com Sklearn

Treina também um modelo de **Regressão Linear Múltipla** com sklearn (Lab 5.1) e compara:
- Os **coeficientes** (pesos) dos dois modelos — devem ser similares!
- O **bias** (intercept)
- As **métricas de desempenho** no conjunto de teste

### 3.5. Visualizações

Cria os seguintes gráficos:
1. **Curva de Aprendizagem:** Loss de treino vs. validação por época
2. **Valores Reais vs. Previstos** no conjunto de teste
3. **Comparação de Coeficientes:** Barplot comparando pesos PyTorch vs. sklearn
4. **Distribuição dos Resíduos**

## 4. Avaliação

Para o modelo neural, calcula:
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- **R²** (Coeficiente de Determinação)

**Pergunta crucial:** Os resultados são idênticos ao modelo sklearn? Se não, por quê?

## 5. Conceitos Importantes

### 5.1. Por que 1 Neurónio = Regressão Linear?

Um neurónio realiza a operação:
```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```

Isto é **exatamente** a equação da regressão linear múltipla!

### 5.2. Hiperparâmetros a Explorar

- **Learning rate:** Muito alto → diverge; muito baixo → treino lento
- **Batch size:** Tamanho do lote para calcular gradientes
- **Número de épocas:** Quantas vezes passar por todo o dataset

### 5.3. Normalização de Features

**Por que é crítico?**
- Diferentes escalas podem fazer gradientes explodirem ou desaparecerem
- Acelera a convergência
- Torna o modelo mais estável

## 6. Produto Final

Um notebook Jupyter ou script Python que contenha:

1. **Código completo e comentado** com:
   - Preparação dos dados em tensors
   - Definição da classe do modelo
   - Loop de treino implementado do zero
   - Avaliação e comparação com sklearn

2. **Visualizações:**
   - Curva de aprendizagem
   - Comparação de coeficientes
   - Scatter plot de previsões

3. **Análise escrita:**
   - Confirmação da equivalência matemática
   - Discussão sobre hiperparâmetros
   - Conclusões sobre quando usar PyTorch vs. sklearn


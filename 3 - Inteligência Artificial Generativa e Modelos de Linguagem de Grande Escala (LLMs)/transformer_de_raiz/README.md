# Um Transformer de raiz, na GPU

Laboratório para estudar **a arquitetura, o treino, a atenção e a evolução da aprendizagem**. Implementa um mini modelo de linguagem que prevê o próximo carácter e gera texto. Não usa pesos pré-treinados nem `nn.Transformer`, `nn.MultiheadAttention` ou atenção pronta.

## Começar neste computador

O ambiente local `.venv` foi preparado com Python 3.12 e PyTorch 2.10.0 + CUDA 12.8. A GPU detetada é uma NVIDIA GeForce RTX 5080 Laptop GPU, com aproximadamente 16 GB. A instalação global de Python/PyTorch não é alterada.

Abre PowerShell **nesta pasta**. Não precisas de ativar o ambiente:

```powershell
# Confirmar que ESTE Python vê a GPU
.\.venv\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

# Testes completos deste laboratório
.\.venv\Scripts\python.exe -m pytest -q

# Treino demonstrativo: 800 atualizações na GPU
.\.venv\Scripts\python.exe train.py --device cuda --steps 800 --output outputs/demo_gpu

# Abrir o relatório com as medições e atenção interativa
Start-Process '.\outputs\demo_gpu\relatorio.html'

# Gerar mais texto com os pesos guardados, sem treinar novamente
.\.venv\Scripts\python.exe generate.py --prompt 'a aluna ' --tokens 200 --temperature 0.8
```

Para preparar o ambiente de novo ou noutro computador Windows com NVIDIA:

```powershell
.\setup_gpu.ps1
```

O script usa `uv`, Python 3.12 e os índices oficiais de PyTorch/PyPI. Requer driver NVIDIA compatível. O comando CUDA é baseado na [documentação oficial para PyTorch 2.10.0](https://pytorch.org/get-started/previous-versions/#v2100). O runtime CUDA vem no pacote; para este treino não é necessário compilar kernels nem instalar o CUDA Toolkit completo.

Alternativa sem `uv`: criar um ambiente com `py -3.12 -m venv .venv` (se essa versão estiver registada), instalar nele `torch==2.10.0` com `--index-url https://download.pytorch.org/whl/cu128`, e depois executar `python -m pip install -r requirements.txt`. Usar sempre o Python da `.venv`.

## O que estás a construir

Imagina completar uma frase consultando apenas as letras já escritas. O Transformer aprende que pistas procurar no contexto; as cabeças de atenção são vários leitores que misturam informação de maneiras diferentes.

```text
texto → IDs por carácter
           │
           ▼
embedding do carácter + embedding da posição     (B, T, 128)
           │
           ▼
┌───────────────────────────────────────────────┐
│ LayerNorm → atenção causal com 4 cabeças       │
│                        + ligação residual     │  × 2 blocos
│ LayerNorm → MLP 128 → 512 → 128                │
│                        + ligação residual     │
└───────────────────────────────────────────────┘
           │
           ▼
LayerNorm → linear → logits                     (B, T, vocabulário)
           │
           ├── treino: cross-entropy com o próximo carácter
           └── geração: softmax → amostrar → acrescentar → repetir
```

- **B:** número de sequências no lote; por defeito, 64.
- **T:** tamanho da janela; por defeito, 64 caracteres.
- **Embedding:** vetor aprendido que representa um carácter; aqui tem 128 componentes.
- **Posição:** outro vetor aprendido, somado ao embedding para representar a ordem.
- **Residual:** soma a entrada à saída de uma transformação; fornece um caminho direto à informação e aos gradientes.
- **MLP:** rede aplicada individualmente a cada posição, depois da mistura de informação feita pela atenção.
- **Logits:** pontuações antes de transformar a previsão em probabilidades.

É uma arquitetura **decoder causal**, inspirada na família GPT. Não é uma reprodução exata de um GPT específico nem o Transformer encoder-decoder completo do artigo original.

## Onde está a atenção

Lê `CausalSelfAttention.forward()` em [model.py](model.py). Para cada cabeça:

1. `Q = X Wq`: cada posição formula uma procura.
2. `K = X Wk`: cada posição oferece uma pista para comparação.
3. `V = X Wv`: cada posição oferece conteúdo a misturar.
4. `Q @ K.transpose(...) / sqrt(head_dim)`: mede compatibilidade entre posições.
5. A máscara coloca `-inf` no futuro. `softmax` transforma esses valores em zero e normaliza cada linha.
6. `weights @ V`: calcula uma mistura do conteúdo das posições permitidas.
7. As cabeças são concatenadas e passam por uma projeção aprendida.

Cada cabeça trabalha com 32 componentes (`128 / 4`). As linhas do mapa são **queries** (quem lê); as colunas são **keys** (de onde lê). A diagonal é permitida porque o carácter atual serve para prever o seguinte. Os mapas capturam as probabilidades antes do dropout.

**Armadilha comum:** permitir acesso ao token seguinte faz a perda parecer excelente, mas deixa o modelo copiar a resposta. O teste de causalidade altera o sufixo e exige que os logits do prefixo permaneçam iguais. Atenção elevada, por si só, também não prova que uma posição causou a resposta: o restante modelo transforma a informação.

## Como se treina

O corpus demonstrativo é **sintético**, criado deterministicamente a partir de combinações de sujeitos, verbos, objetos e complementos. Exemplo: `a aluna estuda a atenção com cuidado.` Não é um corpus representativo de português real.

Dividimos o texto em treino (90%) e validação (10%) **antes** de construir as janelas. No corpus incluído as frases completas não se repetem entre as duas partes, embora a gramática e muitas partes das frases sejam partilhadas.

Uma janela dá a entrada `x` e os alvos `y`, deslocados uma posição:

```text
texto:    a aluna
x:        a alun
y:         aluna
```

O modelo vê os caracteres reais do prefixo durante o treino (teacher forcing). Calculamos cross-entropy sobre todas as posições, fazemos `loss.backward()`, limitamos a norma dos gradientes a 1 e aplicamos `AdamW.step()` com taxa inicial constante de 0,002 e weight decay 0,01. Cada passo é **um lote**, não uma época completa.

O dropout fica ativo no treino e desligado na avaliação. Os lotes de avaliação são fixos e separados do gerador de lotes de treino para comparar as etapas com menos ruído. A validação não calcula gradientes nem atualiza pesos.

## O que aparece nos resultados

Em `outputs/demo_gpu/`:

| Ficheiro | Conteúdo |
|---|---|
| `relatorio.html` | Explicação, medições, curvas, amostras e seleção interativa de etapa/camada/cabeça. Funciona sem internet. |
| `architecture.png` | Diagrama da arquitetura concreta. |
| `loss.png` | Perda e acerto de treino/validação ao longo do treino. |
| `attention.png` | Comparação da atenção inicial/final na última camada. |
| `metrics.csv` | Métricas por etapa, em formato tabular. |
| `samples.json` | Gerações antes, durante e depois do treino. |
| `attention.npz` | Pesos com eixos `(etapa, camada, cabeça, query, key)` e caracteres associados. |
| `summary.json` | Configuração, hardware, parâmetros, tempo, VRAM e resultados. |
| `checkpoint.pt` | Pesos, vocabulário e configuração para inferência. Não inclui estado do otimizador para retomar o treino. |

As amostras usam o mesmo prompt, temperatura e semente. O relatório guarda os dados e imagens dentro do HTML; os restantes ficheiros permitem analisar/exportar os resultados. O tempo medido inclui treino, avaliações, amostras e exportação dos dados, mas exclui a renderização final dos gráficos/HTML.

O código recusa `--device cuda` quando CUDA não está disponível. Para um teste deliberadamente em CPU, usar `--device cpu`; não há fallback silencioso. A métrica de VRAM é o máximo de memória de tensores alocados pelo PyTorch, não a memória total do processo/driver.

## Experiências seguintes

Muda apenas um fator de cada vez e guarda numa pasta diferente:

```powershell
# Contexto mais curto
.\.venv\Scripts\python.exe train.py --device cuda --context-size 16 --output outputs/contexto16

# Inspecionar mais etapas
.\.venv\Scripts\python.exe train.py --device cuda --eval-every 50 --output outputs/mais_etapas

# Texto teu, UTF-8; vocabulário da validação deve existir no treino
.\.venv\Scripts\python.exe train.py --device cuda --corpus 'meu_texto.txt' --output outputs/meu_texto

# Ler um checkpoint dessa experiência
.\.venv\Scripts\python.exe generate.py --checkpoint outputs/meu_texto/checkpoint.pt --prompt 'um início existente no vocabulário'
```

O corpus tem de ser suficientemente longo para as duas partes terem mais caracteres que a janela. A saída de uma nova execução na mesma pasta substitui os resultados anteriores. Tokens desconhecidos no prompt produzem um erro explícito.

Perguntas para acompanhar: a perda de validação acompanha a de treino? O modelo supera a baseline que só conhece frequências de caracteres? As cabeças mudam entre etapas? A geração melhora na gramática limitada das frases? Isso não é evidência de compreensão geral nem de desempenho num corpus novo.

## Mapa do código

1. [model.py](model.py): arquitetura e geração autorregressiva.
2. [train.py](train.py): corpus, lotes, treino, avaliação, amostras e exportação.
3. [report.py](report.py): gráficos e relatório autónomo.
4. [generate.py](generate.py): carregar pesos e gerar sem treino.
5. [tests](tests): causalidade, atenção, gradientes, alinhamento de alvos, aprendizagem e checkpoint.

Os testes automáticos deste laboratório usam CPU para serem rápidos e portáteis; o treino demonstrativo e o teste de execução CUDA verificam a GPU separadamente. A semente melhora a repetibilidade, sem garantir igualdade bit a bit entre hardware/versões.

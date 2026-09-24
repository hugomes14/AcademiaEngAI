# Aula 1 - Introdução à IA, ambientes Python e neurónios simples

## Registo

- Data do registo: 2026-07-14
- Diretório de origem: `Aula1/`
- Módulo: `Modulo 01`
- Laboratório: `Lab01`

## Materiais no repositório

- `Aula1/Modulo 01/Modulo 01/Módulo 01 - Introdução à inteligência artificial e conceitos básicos.pdf`
- `Aula1/Modulo 01/Modulo 01/Módulo 01 - Attention is All You Need.2017.pdf`
- `Aula1/Modulo 01/Modulo 01/Módulo 01 - mad2024.pdf`
- `Aula1/Modulo 01/Modulo 01/Módulo 01 - What is a Digital Twin of the Ocean_-(1080p).mp4`
- `Aula1/Modulo 01/Modulo 01/FirstMark - 2025 MAD (ML-AI-Data) Landscape.url`
- `Aula1/Lab01/Lab01/Lab 01 - Introdução à inteligência artificial e conceitos básicos.html`
- `Aula1/Lab01/Lab01/Lab - Python venv e requirements.txt.html`
- `Aula1/Lab01/Lab01/lab01_ex1.py`
- `Aula1/Lab01/Lab01/lab01_ex2.py`
- `Aula1/Lab01/Lab01/lab01_ex3.py`

## Resumo

A aula introduz conceitos fundamentais de inteligência artificial e mostra como ideias simples podem ser expressas em Python. O laboratório parte de decisões condicionais com `if/else`, avança para uma fórmula linear inspirada em machine learning e termina com a simulação de um neurónio artificial simples.

Também há um laboratório dedicado a ambientes virtuais Python. A ideia principal é perceber que o Python global do sistema e o Python de um projeto podem ser diferentes, e que o `pip` instala bibliotecas no ambiente que estiver ativo. O ficheiro `requirements.txt` regista as dependências do projeto, enquanto a pasta `.venv` guarda o ambiente local que pode ser recriado.

## Conceitos-chave

- Inteligência artificial como área que tenta automatizar decisões, previsões ou classificações.
- Machine learning como abordagem em que uma regra ou modelo usa dados de entrada para produzir uma estimativa.
- Fórmula linear com pesos aplicados a variáveis de entrada.
- Neurónio artificial como combinação de entradas, pesos, bias e função de ativação.
- Elegibilidade mínima antes da classificação: idade igual ou superior a 18 anos e rendimento igual ou superior a 1000 euros.
- Ambientes virtuais Python para isolar dependências por projeto.
- `requirements.txt` como lista reprodutível de bibliotecas necessárias.

## Laboratorio de IA

### Exercício 1 - Decisões com `if/else`

O primeiro exercício recebe idade e rendimento mensal. Se a idade for pelo menos 18 e o rendimento for pelo menos 1000, o programa considera que existe acesso ao crédito; caso contrário, rejeita o acesso.

### Exercício 2 - Fórmula simples de machine learning

O segundo exercício usa a fórmula:

```text
probabilidade_pagamento = 0.01 * idade + 0.0005 * rendimento - 0.5
```

Esta fórmula combina duas variáveis de entrada com pesos fixos e um termo constante. Embora seja simples, já mostra a lógica de um modelo linear: cada variável contribui para uma pontuação final.

### Exercício 3 - Neurónio artificial simples

O terceiro exercício representa a decisão como um neurónio artificial:

```text
ativacao = peso_idade * idade + peso_rendimento * rendimento + bias
```

Com:

```text
peso_idade = 0.01
peso_rendimento = 0.0005
bias = -0.5
```

A função de ativação devolve `1` quando a ativação é maior que `0`, e `0` caso contrário. A implementação em `lab01_ex3.py` usa `numpy` para calcular o produto escalar entre entradas e pesos.

## Laboratorio de ambientes virtuais

### Um ambiente virtual

Fluxo principal:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
deactivate
```

Ideia principal:

```text
antes de ativar .venv -> Python global
depois de ativar .venv -> Python do projeto
```

### Dois ambientes virtuais

O laboratório também mostra que um projeto pode ter mais do que um ambiente virtual, por exemplo:

```text
.venv-web    -> ambiente com requests
.venv-tools  -> ambiente com rich
```

Cada ambiente tem as suas próprias bibliotecas, mas só um fica ativo de cada vez. O `pip` instala sempre no Python ativo no terminal.

## Notas de estudo

- Um neurónio artificial simples é equivalente a uma fórmula linear seguida de uma regra de decisão.
- Os pesos controlam a importância de cada entrada.
- O bias desloca a decisão, tornando o neurónio mais ou menos exigente.
- Antes de aplicar um modelo, podem existir regras mínimas de elegibilidade.
- O `.venv` não deve ser tratado como fonte principal do projeto; pode ser recriado.
- O `requirements.txt` é importante porque descreve as dependências necessárias para reconstruir o ambiente.

## Proximos passos

- Rever a diferença entre uma regra fixa e um modelo treinado.
- Experimentar diferentes valores de pesos e bias no `lab01_ex3.py`.
- Adicionar exemplos de execução com diferentes idades e rendimentos.
- Completar este registo com apontamentos retirados diretamente dos PDFs, se for necessário fazer uma síntese mais detalhada do módulo teórico.

# Dados do laboratório

Os scripts `transformer_treinar.py` e `transformer_treinar_sol.py` usam o conjunto de treino **SQuAD 1.1**, no caminho relativo `conteudo/squad/train-v1.1.json`.

O dataset não é incluído no Git. Pode ser obtido no [site oficial do SQuAD](https://rajpurkar.github.io/SQuAD-explorer/). Para o colocar na pasta esperada, executar em PowerShell a partir desta pasta `lab03`:

```powershell
New-Item -ItemType Directory -Force -Path 'conteudo/squad' | Out-Null
Invoke-WebRequest -Uri 'https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v1.1.json' -OutFile 'conteudo/squad/train-v1.1.json'
```

Os modelos `.pt` produzidos pelo treino também ficam apenas no computador local e não são enviados para o repositório.

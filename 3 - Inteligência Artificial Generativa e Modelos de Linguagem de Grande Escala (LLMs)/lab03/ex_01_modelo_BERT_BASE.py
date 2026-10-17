import torch
# carregar os módulos do framework HuggingFace


# carregar o tokenizador


# carregar o modelo BERT BASE


# frase de Epicteto, um filósofo estoico
s = "a riqueza não consiste em ter grandes posses, mas em ter poucas necessidades"

# frase tokenizada


# saída do modelo
# output_hidden_states=True faz com que tenhamos acesso a todas as camadas na posição 2 da variável de saída


print("Numero de camadas: ", len(out[2]))
print("Numero de lotes: ", len(out[2][0]))
print("Numero de tokens: ", len(out[2][0][0]))
print("Numero de neurónios artificiais: ", len(out[2][0][0][0]))

# Numero de camadas:  13
# Numero de lotes:  1
# Numero de tokens:  26
# Numero de neurónios artificiais:  768
# Ex 2: Fórmula Simples de Machine Learning
# Enunciado:

# Crie um programa em Python que utilize uma fórmula matemática simples para estimar a probabilidade de um cliente pagar um empréstimo, com base na sua idade e rendimento mensal. A fórmula deve ser a seguinte:

# probabilidade_pagamento = 0.01 * idade + 0.0005 * rendimento - 0.5
# O programa deve pedir ao utilizador a sua idade e o seu rendimento mensal, calcular a probabilidade de pagamento usando a fórmula e imprimir o resultado em percentagem.


idade = int(input("Qual é a tua idade? "))

rendimento = int(input("Qual o teu rendimento? "))

if idade >= 18 and rendimento >= 1000:
    print("Tens acesso ao crédito")
    probabilidade_de_pagamento = 0.01*idade + 0.0005*rendimento -0.5

    print(f"A probabilidade de pagamento é {probabilidade_de_pagamento}")
else:
    print("Não tens acesso ao crédito")

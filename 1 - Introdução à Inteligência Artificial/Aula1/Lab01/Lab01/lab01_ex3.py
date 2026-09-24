# Ex 3: Neurónio Artificial Simples
# Enunciado:

# Implemente um programa em Python que simule um neurónio artificial simples para avaliar se um cliente é um bom candidato para um empréstimo, com base na sua idade e rendimento mensal. O neurónio deve utilizar os seguintes pesos e bias:

# peso_idade = 0.01
# peso_rendimento = 0.0005
# bias = -0.5
# O programa deve pedir ao utilizador a sua idade e o seu rendimento mensal, calcular a ativação do neurónio e, com base no resultado, imprimir se o cliente é ou não um bom candidato para o empréstimo. Lembre-se que a função de ativação deve retornar 1 se a ativação for maior que 0, e 0 caso contrário.

# Observação: Em todas as partes, lembre-se que a idade mínima para elegibilidade ao empréstimo é de 18 anos e o rendimento mínimo é de 1000 euros.
import numpy as np



def neuronio_matricial(idade_val, rendimento_val):
    pesos = [0.01, 0.0005]
    bias = -0.5

    if idade_val < 18 or rendimento_val < 1000:
        return None, 'Não tens acesso ao crédito (requisitos mínimos não cumpridos)'


    x = np.array([idade_val, rendimento_val], dtype=float)
    w = np.array(pesos, dtype=float)
    ativacao = float(w.dot(x) + bias)


    classificacao = 1 if ativacao > 0 else 0
    mensagem = 'O cliente é um bom candidato' if classificacao == 1 else 'O cliente não é um bom candidato'
    return ativacao, mensagem


if __name__ == '__main__':
    # Execução interativa mostrando a versão matricial por baixo
    try:
        idade_i = int(input('\n[Matricial] Qual é a tua idade? '))
        rendimento_i = int(input('[Matricial] Qual o teu rendimento? '))
    except Exception:
        print('Entrada inválida.')
    else:
        ativ, msg = neuronio_matricial(idade_i, rendimento_i)
        if ativ is None:
            print(msg)
        else:
            print(f'Ativação (matricial): {ativ:.6f}')
            print(msg)

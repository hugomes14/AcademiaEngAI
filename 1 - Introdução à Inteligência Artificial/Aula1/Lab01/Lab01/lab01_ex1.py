idade = int(input("Qual é a tua idade? "))

rendimento = int(input("Qual o teu rendimento? "))


if idade >= 18 and rendimento >= 1000:
    print("Tens acesso ao crédito")
else:
    print("Não tens acesso ao crédito")
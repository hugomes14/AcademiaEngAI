# Representação simbólica dos fatores e atividades
cidade = "cidade"
praia = "praia"
montanha = "montanha"

sol = "sol"
frio = "frio"
neve = "neve"

historia = "historia"
aventura = "aventura"
relaxamento = "relaxamento"

curta = "curta"
moderada = "moderada"
longa = "longa"

atividade_museus_passeio_curto = "Visitar museus e fazer um passeio guiado pela cidade."
atividade_museus_passeio_moderado = "Visitar museus, fazer um passeio guiado e explorar bairros históricos."
atividade_bicicleta_caminhada = "Fazer um passeio de bicicleta e uma caminhada urbana."
atividade_praia_relaxamento = "Ficar na praia, fazer yoga ao ar livre e relaxar no spa."
atividade_neve_aventura = "Fazer esqui, snowboarding e explorar trilhas de neve."
atividade_default = "Nenhuma atividade recomendada para esta combinação de fatores."

# Definir os fatores atuais
tipo_destino_atual = cidade
preferencia_climatica_atual = sol
interesse_principal_atual = historia
duracao_estadia_atual = moderada

# Inicializar a decisão com a atividade por defeito
decisao = atividade_default

# Aplicar regras simbolicamente
if tipo_destino_atual == cidade and preferencia_climatica_atual == sol and interesse_principal_atual == historia and duracao_estadia_atual == curta:
    decisao = atividade_museus_passeio_curto

if tipo_destino_atual == cidade and preferencia_climatica_atual == sol and interesse_principal_atual == historia and duracao_estadia_atual == moderada:
    decisao = atividade_museus_passeio_moderado

if tipo_destino_atual == cidade and preferencia_climatica_atual == sol and interesse_principal_atual == aventura and duracao_estadia_atual == curta:
    decisao = atividade_bicicleta_caminhada

if tipo_destino_atual == praia and preferencia_climatica_atual == sol and interesse_principal_atual == relaxamento and duracao_estadia_atual == longa:
    decisao = atividade_praia_relaxamento

if tipo_destino_atual == montanha and preferencia_climatica_atual == neve and interesse_principal_atual == aventura and duracao_estadia_atual == longa:
    decisao = atividade_neve_aventura

# Imprimir a decisão final
print(decisao)

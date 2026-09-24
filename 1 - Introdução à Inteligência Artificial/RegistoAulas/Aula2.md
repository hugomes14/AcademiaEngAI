# Aula 2 - Abordagem simbólica e sistemas baseados em regras

## Registo

- Data do registo: 2026-07-14
- Diretório de origem: `Aula2/`
- Módulo: `Módulo 02`
- Laboratório: `Lab02`

## Materiais no repositório

- `Aula2/Módulo 02/Módulo 02/Módulo 02 - Principais abordagens em IA - abordagem simbólica.pdf`
- `Aula2/Módulo 02/Módulo 02/Módulo 02 - Exemplo 1 Escolha da Roupa.html`
- `Aula2/Módulo 02/Módulo 02/Módulo 02 - Exemplo 2 Planear a Refeição.html`
- `Aula2/Módulo 02/Módulo 02/Módulo 02 - Exemplo 3 Seleção de Atividade.html`
- `Aula2/Módulo 02/Módulo 02/Módulo 02 - Exemplo 4 Ligação das Luzes.html`
- `Aula2/Módulo 02/Módulo 02/Módulo 02 - Exemplo 5 Manutenção do Carro.html`
- `Aula2/Lab02/Lab02/Lab 02 - Principais abordagens em IA - abordagem simbólica.html`
- `Aula2/Lab02/Lab02/lab02_ex1.py`
- `Aula2/Lab02/Lab02/lab02_ex2.py`
- `Aula2/Lab02/Lab02/run_drone.py`

## Resumo

Esta aula apresenta a abordagem simbólica em inteligência artificial. Em vez de aprender a partir de dados, o sistema representa factos, estados e decisões através de símbolos e regras explícitas. As decisões são explicáveis porque é possível identificar a regra aplicada, as condições que a ativaram e o estado produzido.

O laboratório começa com um recomendador de atividades baseado em condições. Depois evolui para um drone simulado em `pygame`, controlado por uma base de conhecimento e um motor de inferência. O sistema prevê colisões, calcula desvios, move o drone e o obstáculo, atualiza a navegação e regista cada decisão importante num ficheiro de log.

## Conceitos-chave

- IA simbólica: abordagem baseada em símbolos, factos e regras definidas por pessoas.
- Base de dados: conjunto de factos estáticos e dinâmicos sobre o ambiente.
- Base de regras: conjunto de funções ou condições que transforma o estado do sistema.
- Motor de inferência: componente que aplica as regras numa ordem definida.
- Representação do conhecimento: uso de variáveis e estados legíveis, como `estado_drone`, `destino` e `evitando_obstaculo`.
- Explicabilidade: registo da regra ativada, estado anterior, estado novo e justificação.
- Planeamento reativo: previsão de colisões e cálculo de uma rota alternativa antes do impacto.

## Laboratório de IA simbólica

### Exercício 1 - Recomendação de atividades

O primeiro exercício representa fatores como destino, clima, interesse e duração da estadia através de símbolos. Regras com `if` combinam esses fatores para escolher uma atividade.

Exemplo de raciocínio:

```text
cidade + sol + história + estadia moderada
-> visitar museus, fazer um passeio guiado e explorar bairros históricos
```

Quando nenhuma regra corresponde à combinação atual, o programa devolve uma recomendação por defeito. Este exercício mostra como uma decisão pode ser construída inteiramente por regras explícitas.

### Exercício 2 - Drone baseado em regras

O segundo exercício organiza o sistema em duas partes:

```text
Base de conhecimento
-> dados do ambiente e estado do drone

Motor de inferência
-> sequência de regras que atualiza o estado
```

A base de dados inclui dimensões da janela, locais da rota, configuração do obstáculo, posição do drone, destino, estado de movimento e informação sobre desvios. A rota principal segue a sequência:

```text
Base -> A -> B -> C -> D -> Base
```

### Regras implementadas

- `regra_mover_obstaculo`: desloca o obstáculo verticalmente e inverte a direção nos limites da janela.
- `verificar_colisao`: usa retângulos `pygame.Rect` para detetar sobreposição entre o drone e o obstáculo.
- `prever_colisao`: avalia pontos do segmento entre a posição atual e o destino para antecipar uma colisão.
- `calcular_rota_alternativa`: calcula um ponto de desvio lateral ao obstáculo.
- `regra_evitar_obstaculo`: guarda o destino original, ativa a rota alternativa e retoma o destino quando o desvio termina.
- `regra_mover_drone`: aproxima o drone do destino a uma velocidade de 5 px por ciclo.
- `regra_atualizar_estado_navegacao`: avança para o próximo ponto da rota após completar um segmento.
- `regra_verificar_colisao_real`: altera o estado para `Colisão` quando existe impacto efetivo.

### Ciclo de decisão

O motor de inferência aplica as regras nesta ordem:

```text
prever e evitar obstáculo
-> mover drone
-> atualizar navegação
-> mover obstáculo
-> verificar colisão real
```

Se o estado do drone for `Colisão`, o motor deixa de aplicar novas regras. Esta ordem separa a previsão de risco da confirmação final de uma colisão.

## Explicabilidade e logs

O sistema usa `log_explicabilidade` para criar um registo de decisões. Cada entrada inclui:

- data e hora;
- regra ativada;
- estado anterior;
- estado novo;
- justificação.

São registados, entre outros, os eventos de inicialização, início de desvio, retoma da rota original, atualização de navegação e colisão real.

## Notas de estudo

- Regras simbólicas são fáceis de inspecionar e justificar, mas precisam de ser definidas e mantidas manualmente.
- A base de conhecimento reúne tanto constantes do ambiente como factos que mudam durante a execução.
- `prever_colisao` é uma decisão preventiva; `regra_verificar_colisao_real` confirma o resultado depois das atualizações do ciclo.
- O estado `Colisão` bloqueia novos movimentos e funciona como estado terminal da simulação.
- A rota alternativa é temporária: o sistema conserva `destino_original` para poder regressar ao plano inicial.

## Próximos passos

- Executar `run_drone.py` num ambiente com `pygame` instalado e observar o comportamento do drone.
- Testar posições, velocidades e dimensões diferentes para o obstáculo.
- Melhorar a previsão aumentando o número de pontos analisados no trajeto.
- Comparar este sistema de regras com uma solução baseada em machine learning, onde a política de desvio seria aprendida a partir de dados.

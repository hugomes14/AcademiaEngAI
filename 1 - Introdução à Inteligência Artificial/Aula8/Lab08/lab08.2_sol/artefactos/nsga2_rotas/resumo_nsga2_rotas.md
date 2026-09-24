# Otimização Multiobjetivo de Rotas

## Frente de Pareto

| rota                              | waypoints_intermedios   |   tempo_voo |   consumo_combustivel |
|:----------------------------------|:------------------------|------------:|----------------------:|
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |
| 0 -> 2 -> 1 -> 10 -> 3 -> 9 -> 11 | [2, 1, 10, 3, 9]        |     106.672 |               95.6748 |

## Interpretação

- Cada ponto da Frente de Pareto representa uma rota não dominada.
- Uma rota não dominada não pode melhorar o tempo sem piorar o consumo, ou vice-versa.
- O gráfico mostra o compromisso entre rapidez e eficiência energética.

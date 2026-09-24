"""
07 — Relatório automático em Markdown.

Objetivo:
- Reunir resultados do Bandit e do Q-Learning.
- Criar RELATORIO_FINAL.md com tabelas, imagens e interpretação.
"""

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
TABELAS_DIR = BASE_DIR / "artefactos" / "tabelas"
IMAGENS_DIR = BASE_DIR / "artefactos" / "imagens"
RELATORIOS_DIR = BASE_DIR / "artefactos" / "relatorios"
RELATORIOS_DIR.mkdir(parents=True, exist_ok=True)

print("\n" + "=" * 80)
print("LAB 08.1 — RELATÓRIO AUTOMÁTICO")
print("=" * 80)

bandit_resumo_path = TABELAS_DIR / "bandit_resumo.csv"
avaliacao_path = TABELAS_DIR / "qlearning_avaliacao_metricas.csv"
historico_path = TABELAS_DIR / "qlearning_historico_treino.csv"

bandit_resumo = pd.read_csv(bandit_resumo_path) if bandit_resumo_path.exists() else pd.DataFrame()
avaliacao = pd.read_csv(avaliacao_path) if avaliacao_path.exists() else pd.DataFrame()
historico = pd.read_csv(historico_path) if historico_path.exists() else pd.DataFrame()

if not historico.empty:
    recompensa_final = historico["recompensa_media_movel_100"].dropna().tail(1).iloc[0]
    sucesso_final = historico["taxa_sucesso_movel_100"].dropna().tail(1).iloc[0]
else:
    recompensa_final = None
    sucesso_final = None

relatorio = []
relatorio.append("# Relatório Final — Lab 08.1 Aprendizagem por Reforço\n")
relatorio.append("## 1. Introdução\n")
relatorio.append(
    "Este laboratório demonstra dois cenários clássicos de Aprendizagem por Reforço: "
    "um problema Multi-Armed Bandit para seleção de rotas de voo e um Grid World para "
    "aprendizagem de uma política de navegação com Q-Learning.\n"
)

relatorio.append("## 2. Ambiente e Agente\n")
relatorio.append(
    "- **Bandit:** cada rota representa uma ação com recompensa incerta. "
    "O agente procura maximizar a recompensa ao longo do tempo.\n"
    "- **Grid World:** cada estado representa uma posição numa grelha 10x10. "
    "O agente aprende uma política que conduz ao destino e evita obstáculos.\n"
    "- **Q-Learning:** usa uma Q-table para estimar o valor de cada par estado-ação.\n"
)

relatorio.append("## 3. Exploração vs. Explotação no Bandit\n")
if not bandit_resumo.empty:
    relatorio.append(bandit_resumo.to_markdown(index=False))
    relatorio.append("\n")
else:
    relatorio.append("A tabela de resultados do Bandit não foi encontrada.\n")

relatorio.append("![Recompensa média acumulada](../imagens/bandit_recompensa_media_acumulada.png)\n")
relatorio.append("![Regret acumulado](../imagens/bandit_regret_acumulado.png)\n")

relatorio.append(
    "O regret acumulado mede a diferença entre a recompensa obtida e a recompensa "
    "que seria esperada se o agente escolhesse sempre a melhor rota real. Um regret "
    "mais baixo indica aprendizagem mais eficiente.\n"
)

relatorio.append("## 4. Treino com Q-Learning\n")
relatorio.append(
    "O treino usa uma política epsilon-greedy. No início, `epsilon` é alto para favorecer "
    "exploração. Ao longo dos episódios, `epsilon` desce para favorecer a exploração da "
    "melhor política conhecida.\n"
)
if recompensa_final is not None:
    relatorio.append(f"- Recompensa média móvel final: **{recompensa_final:.3f}**\n")
if sucesso_final is not None:
    relatorio.append(f"- Taxa de sucesso móvel final: **{sucesso_final:.2%}**\n")

relatorio.append("![Curva de aprendizagem](../imagens/qlearning_curva_aprendizagem.png)\n")
relatorio.append("![Taxa de sucesso](../imagens/qlearning_taxa_sucesso.png)\n")
relatorio.append("![Decaimento de epsilon](../imagens/qlearning_epsilon.png)\n")

relatorio.append("## 5. Avaliação da Política\n")
if not avaliacao.empty:
    relatorio.append(avaliacao.to_markdown(index=False))
    relatorio.append("\n")
else:
    relatorio.append("A tabela de avaliação não foi encontrada.\n")

relatorio.append(
    "A política treinada deve superar o agente aleatório em recompensa média e taxa de sucesso. "
    "A avaliação não usa exploração intencional; por isso, mede a qualidade da política aprendida.\n"
)

relatorio.append("## 6. Visualização da Política\n")
relatorio.append("![Função valor](../imagens/qlearning_funcao_valor_heatmap.png)\n")
relatorio.append("![Política ótima](../imagens/qlearning_politica_setas.png)\n")
relatorio.append(
    "A função valor mostra os estados com maior valor estimado. A política com setas mostra a "
    "ação preferida em cada célula normal. Obstáculos e destino aparecem assinalados no mapa.\n"
)

relatorio.append("## 7. Conceitos-Chave\n")
relatorio.append(
    "- **Estado:** situação atual do agente.\n"
    "- **Ação:** decisão possível em cada estado.\n"
    "- **Recompensa:** sinal que orienta a aprendizagem.\n"
    "- **Alpha:** controla quanto a nova informação altera a Q-table.\n"
    "- **Gamma:** controla a importância de recompensas futuras.\n"
    "- **Epsilon:** controla o equilíbrio entre exploração e explotação.\n"
    "- **Equação de Bellman:** atualiza o valor esperado de cada ação em cada estado.\n"
)

relatorio.append("## 8. Conclusões e Recomendações\n")
relatorio.append(
    "A Aprendizagem por Reforço é adequada para problemas de decisão sequencial, nos quais "
    "o efeito de uma ação só se torna claro depois de vários passos. No Bandit, o foco está "
    "na seleção repetida da melhor rota. No Grid World, o foco está na aprendizagem de uma "
    "política completa de navegação.\n\n"
    "Recomendações para aprofundamento:\n"
    "- Testar diferentes recompensas para analisar *reward shaping*.\n"
    "- Alterar obstáculos no Grid World para avaliar robustez.\n"
    "- Comparar Q-Learning com SARSA.\n"
    "- Explorar DQN quando o espaço de estados deixa de caber numa Q-table.\n"
)

relatorio_path = RELATORIOS_DIR / "RELATORIO_FINAL.md"
relatorio_path.write_text("\n".join(relatorio), encoding="utf-8")

print(f"Relatório criado: {relatorio_path}")

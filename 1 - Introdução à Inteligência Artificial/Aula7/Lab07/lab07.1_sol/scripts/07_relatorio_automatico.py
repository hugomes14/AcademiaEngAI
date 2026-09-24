# 07_relatorio_automatico.py
# Lab 07.1 — Clustering: Descoberta de Perfis Operacionais de Voo
# Etapa 7: Relatório automático em Markdown
#
# Objetivo pedagógico:
# - Consolidar resultados, gráficos e interpretações num relatório final.
# - Explicar as decisões principais de clustering.
# - Apresentar perfis operacionais de voo.

from pathlib import Path
import json
import warnings

import pandas as pd

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1] if "__file__" in globals() else Path(".")
TABLE_DIR = BASE_DIR / "outputs" / "tables"
FIG_DIR = BASE_DIR / "outputs" / "figures"
MODEL_DIR = BASE_DIR / "outputs" / "models"
REPORT_DIR = BASE_DIR / "outputs" / "reports"

REPORT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("LAB 07.1 — ETAPA 7: RELATÓRIO AUTOMÁTICO")
print("=" * 80)

relatorio_path = REPORT_DIR / "RELATORIO_FINAL.md"

eda_desc = pd.read_csv(TABLE_DIR / "eda_descritiva.csv", index_col=0) if (TABLE_DIR / "eda_descritiva.csv").exists() else pd.DataFrame()
outliers = pd.read_csv(TABLE_DIR / "eda_outliers_iqr.csv") if (TABLE_DIR / "eda_outliers_iqr.csv").exists() else pd.DataFrame()
elbow = pd.read_csv(TABLE_DIR / "elbow_metricas_kmeans.csv") if (TABLE_DIR / "elbow_metricas_kmeans.csv").exists() else pd.DataFrame()
metricas = pd.read_csv(TABLE_DIR / "metricas_clustering.csv") if (TABLE_DIR / "metricas_clustering.csv").exists() else pd.DataFrame()
perfil = pd.read_csv(TABLE_DIR / "perfil_clusters_medias.csv") if (TABLE_DIR / "perfil_clusters_medias.csv").exists() else pd.DataFrame()
personas = pd.read_csv(TABLE_DIR / "personas_clusters.csv") if (TABLE_DIR / "personas_clusters.csv").exists() else pd.DataFrame()

if (MODEL_DIR / "k_recomendado.json").exists():
    with open(MODEL_DIR / "k_recomendado.json", "r", encoding="utf-8") as f:
        k_info = json.load(f)
    k_recomendado = k_info.get("k_recomendado", "n/d")
else:
    k_recomendado = "n/d"

if (MODEL_DIR / "melhor_modelo_clustering.txt").exists():
    melhor_modelo = (MODEL_DIR / "melhor_modelo_clustering.txt").read_text(encoding="utf-8").strip()
else:
    melhor_modelo = "n/d"

conteudo = []
conteudo.append("# Relatório Final — Lab 07.1: Clustering — Perfis Operacionais de Voo\n")
conteudo.append("## 1. Introdução\n")
conteudo.append(
    "Este relatório apresenta uma análise não supervisionada de voos com base em métricas agregadas de telemetria. "
    "O objetivo não é prever uma variável-alvo, mas descobrir perfis operacionais com características semelhantes.\n"
)

conteudo.append("## 2. Dataset\n")
conteudo.append("O dataset usado foi `voos_telemetria_completa.csv`, com as seguintes variáveis:\n")
conteudo.append("- `duracao_voo_min`\n")
conteudo.append("- `distancia_percorrida_km`\n")
conteudo.append("- `altitude_maxima_m`\n")
conteudo.append("- `velocidade_media_kmh`\n")
conteudo.append("- `consumo_combustivel_litros`\n")
conteudo.append("- `variacao_vertical_total_m`\n")

conteudo.append("\n## 3. Análise Exploratória\n")
conteudo.append(
    "A análise exploratória confirmou que as variáveis têm escalas muito diferentes. "
    "Este ponto é crítico, porque K-Means, DBSCAN e clustering hierárquico usam distâncias entre observações.\n"
)

if not eda_desc.empty:
    conteudo.append("\n### Estatística descritiva\n")
    conteudo.append(eda_desc.to_markdown(floatfmt=".3f"))
    conteudo.append("\n")

if not outliers.empty:
    conteudo.append("\n### Outliers pelo método IQR\n")
    conteudo.append(outliers.to_markdown(index=False, floatfmt=".3f"))
    conteudo.append("\n")

conteudo.append("\n### Figuras da EDA\n")
conteudo.append("- Histogramas e boxplots: `outputs/figures/eda/`\n")
conteudo.append("- Pairplot: `outputs/figures/eda/pairplot_variaveis.png`\n")
conteudo.append("- Correlações: `outputs/figures/eda/heatmap_correlacoes.png`\n")

conteudo.append("\n## 4. Pré-processamento\n")
conteudo.append(
    "Foi aplicado `StandardScaler` ao dataset completo, porque não existe variável-alvo nem divisão treino/teste. "
    "O escalonamento impede que variáveis com unidades maiores, como altitude ou variação vertical, dominem as distâncias.\n"
)

conteudo.append("\nArtefactos principais:\n")
conteudo.append("- `outputs/tables/dados_limpos.csv`\n")
conteudo.append("- `outputs/tables/dados_processados_sem_escala.csv`\n")
conteudo.append("- `outputs/tables/dados_escalonados.csv`\n")
conteudo.append("- `outputs/models/standard_scaler.pkl`\n")

conteudo.append("\n## 5. Método Elbow e Escolha de k\n")
conteudo.append(f"O `k` recomendado automaticamente pela Silhouette foi: **{k_recomendado}**.\n")

if not elbow.empty:
    conteudo.append("\n### Métricas por k\n")
    conteudo.append(elbow.to_markdown(index=False, floatfmt=".4f"))
    conteudo.append("\n")

conteudo.append("\nFiguras geradas:\n")
conteudo.append("- `outputs/figures/elbow/metodo_elbow_inercia.png`\n")
conteudo.append("- `outputs/figures/elbow/silhouette_por_k.png`\n")
conteudo.append("- `outputs/figures/elbow/davies_bouldin_por_k.png`\n")
conteudo.append("- `outputs/figures/elbow/calinski_harabasz_por_k.png`\n")

conteudo.append("\n## 6. Modelos de Clustering\n")
conteudo.append("Foram treinados três modelos:\n")
conteudo.append("- K-Means\n")
conteudo.append("- DBSCAN\n")
conteudo.append("- AgglomerativeClustering\n")

conteudo.append("\nAs métricas internas usadas foram:\n")
conteudo.append("- **Silhouette Score**: maior é melhor.\n")
conteudo.append("- **Davies-Bouldin Index**: menor é melhor.\n")
conteudo.append("- **Calinski-Harabasz Index**: maior é melhor.\n")

if not metricas.empty:
    conteudo.append("\n### Tabela comparativa\n")
    conteudo.append(metricas.to_markdown(index=False, floatfmt=".4f"))
    conteudo.append("\n")

conteudo.append(f"\nMelhor modelo por Silhouette: **{melhor_modelo}**.\n")

conteudo.append("\n## 7. Visualização PCA\n")
conteudo.append(
    "A PCA foi usada apenas para visualização em 2D. As métricas foram calculadas no espaço completo escalonado.\n"
)
conteudo.append("\nFiguras principais:\n")
conteudo.append("- `outputs/figures/pca/pca_melhor_modelo.png`\n")
conteudo.append("- `outputs/figures/pca/pca_kmeans.png`\n")
conteudo.append("- `outputs/figures/pca/pca_agglomerative.png`\n")
conteudo.append("- `outputs/figures/pca/pca_dbscan.png`\n")

conteudo.append("\n## 8. DBSCAN e Outliers\n")
conteudo.append(
    "O DBSCAN permite identificar observações como ruído (`label = -1`). "
    "Isto é útil quando alguns voos não encaixam bem nos perfis dominantes.\n"
)
conteudo.append("\nFigura principal:\n")
conteudo.append("- `outputs/figures/perfil_clusters/pca_dbscan_ruido_destacado.png`\n")

conteudo.append("\n## 9. Perfil dos Clusters\n")
conteudo.append(
    "O profiling foi feito com os dados originais, não escalonados, para manter as unidades interpretáveis.\n"
)

if not perfil.empty:
    conteudo.append("\n### Médias por cluster\n")
    conteudo.append(perfil.to_markdown(index=False, floatfmt=".3f"))
    conteudo.append("\n")

if not personas.empty:
    conteudo.append("\n### Personas sugeridas\n")
    conteudo.append(personas.to_markdown(index=False))
    conteudo.append("\n")

conteudo.append("\nFiguras principais:\n")
conteudo.append("- `outputs/figures/perfil_clusters/perfil_clusters_medias_normalizadas.png`\n")
conteudo.append("- `outputs/figures/perfil_clusters/heatmap_perfil_clusters.png`\n")

conteudo.append("\n## 10. Conclusões\n")
conteudo.append(
    "O laboratório demonstra que clustering não termina no treino do modelo. "
    "A etapa decisiva é interpretar os grupos e confirmar se têm utilidade operacional.\n"
)
conteudo.append("\nPontos principais:\n")
conteudo.append("- O escalonamento é obrigatório em algoritmos baseados em distância.\n")
conteudo.append("- A escolha de `k` deve combinar métricas, gráficos e interpretação de negócio.\n")
conteudo.append("- Outliers podem alterar centróides no K-Means e devem ser analisados.\n")
conteudo.append("- DBSCAN é útil para detetar observações atípicas, mas é sensível aos parâmetros `eps` e `min_samples`.\n")
conteudo.append("- A PCA ajuda a comunicar resultados, mas não substitui as métricas calculadas no espaço original de features escalonadas.\n")

conteudo.append("\n## 11. Recomendações\n")
conteudo.append("- Testar outros valores de `k` próximos do recomendado.\n")
conteudo.append("- Comparar `StandardScaler` com `RobustScaler` se houver muitos outliers.\n")
conteudo.append("- Validar os perfis com especialistas de operação de voo.\n")
conteudo.append("- Experimentar GMM caso se suspeite de clusters elípticos.\n")
conteudo.append("- Repetir a análise com novas variáveis de telemetria, se existirem.\n")

conteudo.append("\n## 12. Referências\n")
conteudo.append("- scikit-learn: KMeans, DBSCAN, AgglomerativeClustering, PCA.\n")
conteudo.append("- Métricas internas de clustering: Silhouette, Davies-Bouldin, Calinski-Harabasz.\n")

relatorio_path.write_text("\n".join(conteudo), encoding="utf-8")
(BASE_DIR / "RELATORIO_FINAL.md").write_text("\n".join(conteudo), encoding="utf-8")

print(f"\n[OK] Relatório final criado em: {relatorio_path}")

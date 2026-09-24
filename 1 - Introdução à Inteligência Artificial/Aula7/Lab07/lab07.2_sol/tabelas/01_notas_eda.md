# Notas da EDA — Lab 07.2

- Dataset com 600 linhas e 51 colunas.
- Foram detetadas 50 features numéricas.
- A coluna `tipo_manobra` existe e será usada apenas para visualização e LDA.
- O PCA é sensível à escala, porque procura direções de maior variância.
- Sem escalonamento, sensores com valores absolutos maiores podem dominar os componentes principais.
- Outliers podem distorcer a variância e, por isso, alterar os componentes principais.
- Correlações fortes entre sensores indicam redundância potencial, que o PCA pode comprimir.
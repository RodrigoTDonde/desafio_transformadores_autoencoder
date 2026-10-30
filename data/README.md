# Dados do exercício de transformadores

## Origem verificada

As sete variáveis numéricas de todas as 17.348 linhas de `transformers.csv` coincidem com o dataset público ETTh1 nas respectivas datas, com tolerância absoluta de 1e-5. A diferença máxima foi 7,11e-15. O arquivo público contém 17.420 linhas e não contém a coluna `class` do exercício.

- Fonte: [ETDataset, de Haoyi Zhou e colaboradores](https://github.com/zhouhaoyi/ETDataset).
- Arquivo público: [ETT-small/ETTh1.csv](https://github.com/zhouhaoyi/ETDataset/blob/main/ETT-small/ETTh1.csv).
- Referência solicitada pela fonte: Zhou et al. (2021), *Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting*, AAAI 2021.
- Termos publicados na fonte: [CC BY-ND 4.0](https://github.com/zhouhaoyi/ETDataset/blob/main/LICENSE). Esses termos não são uma licença para todo o código deste projeto.
- Hashes e contagens da comparação: [proveniencia-verificada.json](proveniencia-verificada.json).

## Coluna adicional e publicação

O repositório acadêmico usa a coluna `class` como rótulo de normalidade/falha, mas não documenta como ela foi criada. A coincidência numérica com ETTh1 não comprova a procedência desses rótulos nem que indiquem falhas reais. Métricas que usam `class` devem ser interpretadas como resultados do exercício, sem alegar validação de falhas reais.

Esta cópia foi tornada pública por solicitação do titular do portfólio em 30/10/2026. A publicação não esclarece a origem dos rótulos nem altera os termos da fonte, que continuam registrados acima. Não se deve substituir silenciosamente o CSV pelo ETTh1 público: ele tem quantidade de linhas diferente e não oferece `class`, portanto não reproduz diretamente as métricas históricas.

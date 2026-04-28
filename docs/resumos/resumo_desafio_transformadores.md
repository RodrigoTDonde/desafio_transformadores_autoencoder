# Resumo — Desafio de Transformadores no Setor Elétrico

> **Fonte:** [docs/referencias/desafio_transformadores.pdf](../referencias/desafio_transformadores.pdf)
> **Dataset:** [data/transformers.csv](../../data/transformers.csv)

---

## 1. Contexto do Problema

Os transformadores são componentes críticos no sistema elétrico, responsáveis por
garantir o fornecimento seguro e estável de energia. A falha em um transformador
pode causar interrupções significativas, danos a equipamentos e custos elevados
de manutenção.

A **temperatura do óleo** é um dos principais indicadores de saúde do
transformador, refletindo a eficiência térmica e o nível de carga aplicado ao
sistema. Entretanto, prever o comportamento futuro dessa temperatura e
identificar situações anômalas é um grande desafio, pois os dados são
influenciados por múltiplas variáveis externas.

Para este desafio, trabalha-se com o dataset `transformadores.csv`, que contém
**medições reais de cargas e temperatura do óleo de transformadores coletadas a
cada hora, ao longo de dois anos**. O objetivo é aplicar técnicas de
aprendizado profundo (Deep Learning) para:

- detecção de anomalias,
- previsão temporal,
- classificação de condições operacionais.

---

## 2. Dataset

| Coluna                                 | Descrição                                          |
| -------------------------------------- | -------------------------------------------------- |
| `date`                                 | Data/hora da medição                               |
| `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL` | Cargas elétricas em diferentes regiões     |
| `OT`                                   | Temperatura do óleo do transformador               |
| `class`                                | Indicador binário de condição (0 = normal, 1 = falha) |

- **Granularidade temporal:** horária
- **Horizonte:** 2 anos de medições

---

## 3. Tarefas do Desafio

### 3.1 Detecção de Anomalias — Autoencoder denso

- **Objetivo:** detectar comportamentos anormais nos dados operacionais.
- **Descrição:** utilizar um Autoencoder denso para reconstruir as entradas e
  calcular o erro de reconstrução.
- **Threshold:** definir um limiar para identificar anomalias, por exemplo,
  usando o **percentil 99** do erro.
- **Entrada:** todas as features numéricas, exceto a variável `class`.
- **Saída:** gráficos com as anomalias em vermelho ao longo do tempo; decidir,
  com base na visualização, se será necessário removê-las ou não.

### 3.2 Predição Multivariada — LSTM

- **Objetivo:** prever a temperatura futura do óleo (`OT`) a partir do
  histórico das demais variáveis.
- **Descrição:** usar uma LSTM multivariada com **janelas deslizantes de tempo**
  (por exemplo, 24 passos passados).
- **Horizonte de previsão:** os próximos **4 passos** da temperatura do óleo.
- **Avaliação:** gráficos comparativos entre valores reais e previstos.
- **Entrada:** todas as features numéricas, exceto `class`.
- **Saída:** valores previstos de `OT` para os próximos 4 instantes.

### 3.3 Classificação de Condição Operacional — Deep MLP

- **Objetivo:** classificar o estado operacional do transformador em condição
  **normal (0)** ou **falha/potencial risco (1)**.
- **Descrição:** usar um perceptron multicamadas profundo (Deep MLP) com
  camadas densas e **dropout**.
- **Treino:** todas as variáveis numéricas como entrada e a coluna `class` como
  rótulo.
- **Avaliação:** relatório de classificação com **acurácia, precisão, recall e
  F1-score**.
- **Entrada:** todas as features numéricas.
- **Saída:** predição binária (0 = normal, 1 = falha).

---

## 4. Entregas Esperadas

- [ ] Autoencoder treinado + gráfico temporal com anomalias destacadas e
      decisão sobre remoção.
- [ ] LSTM multivariada com previsão de 4 passos à frente para `OT` e gráfico
      real × previsto.
- [ ] Deep MLP com relatório de classificação (acurácia, precisão, recall,
      F1-score).

---

## 5. Próximos Passos

1. Inspecionar `data/transformers.csv` (estatísticas, valores faltantes,
   distribuição de `class`).
2. Implementar as três tarefas em `notebooks/` e/ou `src/`.
3. Documentar resultados e gráficos de cada modelo.

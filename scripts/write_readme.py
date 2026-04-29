"""Reescreve o README.md em UTF-8 (alguns linters/ferramentas estão salvando-o em UTF-16)."""

from pathlib import Path

CONTENT = """# Desafio — Transformadores no Setor Elétrico

> **Projeto acadêmico — Pós-graduação SENAI em Deep Learning (Aula 8: Autoencoders)**
> Aplicação de Deep Learning para monitoramento de transformadores: detecção
> de anomalias, previsão temporal e classificação de condições operacionais.

---

## Contexto

Transformadores são componentes críticos do sistema elétrico. A **temperatura
do óleo** é um dos principais indicadores de saúde térmica do equipamento, e
falhas geram interrupções, danos a equipamentos e custos elevados de
manutenção.

O dataset `data/transformers.csv` contém **17.348 medições horárias reais**
coletadas ao longo de **2 anos** (julho/2016 a junho/2018), com 7 variáveis
numéricas (cargas em diferentes regiões + temperatura do óleo) e um rótulo
binário de condição (`class`: 0 = normal, 1 = falha — apenas ~1% dos
registros).

## Perguntas de negócio

| # | Pergunta | Técnica utilizada |
|---|---|---|
| 1 | Quais momentos apresentam **comportamento operacional anômalo**? | **Autoencoder** denso + erro de reconstrução |
| 2 | Qual será a **temperatura do óleo nas próximas 4 horas**? | **LSTM** multivariada (janela de 24h) |
| 3 | O instante atual é **normal ou de falha**? | **Deep MLP** com `class_weight` balanceado |

## Resultados

| Tarefa | Modelo | Métricas |
|---|---|---|
| Detecção de anomalias | Autoencoder denso (`7 → 4 → 2 → 4 → 7`) | 1.01 % de anomalias detectadas (threshold p99) |
| Previsão de OT (4 h) | LSTM(64) → Dense(32) → Dense(4) | MAE ≈ **0.78 ºC**, RMSE ≈ **1.08 ºC** |
| Classificação binária | MLP(64→32→1) com dropout 0.3 | F1 ≈ 0.11, Recall ≈ 0.26 (threshold calibrado em validação) |

> O baixo F1 do MLP reflete o forte **desbalanceamento (~1 % de falhas)** e a
> mudança de distribuição entre treino (0.5 %) e teste (3 %) — limitação real
> do problema, documentada na fase Deployment do notebook.

## Estrutura do projeto

```
pos_senai_deep_learning_desafio_transformadores_autoencoder/
├── data/
│   └── transformers.csv                # 17.348 medições horárias (2016–2018)
├── docs/
│   ├── referencias/                    # Material de aula (PDFs e notebook base)
│   └── resumos/                        # Resumos analíticos do contexto
│       ├── resumo_desafio_transformadores.md
│       ├── resumo_aula_8_autoencoders.md
│       └── tecnicas_notebook_referencia.md
├── notebooks/
│   └── desafio_transformadores.ipynb   # Notebook executável (CRISP-DM)
├── reports/
│   └── images/                         # 13 figuras geradas (PNG)
├── scripts/
│   └── build_notebook.py               # Reconstrói o notebook idempotentemente
├── src/
│   ├── __init__.py
│   └── utils.py                        # Funções auxiliares modularizadas
├── requirements.txt
└── README.md
```

## Metodologia

O notebook segue **CRISP-DM** ponta a ponta, com cada subtópico estruturado em
**Objetivo / Técnicas / Saída** e fechado por uma célula de **Diagnóstico,
Highlights e Decisão para a próxima etapa**:

1. **Business Understanding** — perguntas de negócio e critérios de sucesso.
2. **Data Understanding** — carga, descritiva, série temporal, desbalanceamento, correlações e *deep dive* na temperatura do óleo (OT).
3. **Data Preparation** — split temporal cronológico, `StandardScaler`, janelas deslizantes.
4. **Modeling** — Autoencoder, LSTM e Deep MLP.
5. **Evaluation** — métricas e plots por modelo + síntese cruzada.
6. **Deployment** — respostas consolidadas, recomendações e limitações.

### Boas práticas aplicadas

- **Split cronológico** (sem shuffle) para evitar data leakage em série temporal.
- **`StandardScaler` ajustado apenas no treino**.
- **Autoencoder treinado somente em `class==0`** (premissa do método não supervisionado).
- **`class_weight="balanced"` + calibração de threshold em validação** para o MLP — sem técnicas avançadas (SMOTE etc.).
- **EarlyStopping** para evitar overfitting.
- **Seeds fixadas** (`numpy`, `random`, `tensorflow`).

## Como executar

### 1. Pré-requisitos

- Python 3.10+ (testado com 3.12)
- Sistema operacional com TensorFlow CPU disponível (Windows/Linux/macOS)

### 2. Instalação

```bash
# Criar e ativar um ambiente virtual (opcional, recomendado)
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# Linux/macOS
source .venv/bin/activate

# Instalar dependências
pip install -r requirements.txt
```

### 3. Executar o notebook

```bash
# Opção A: Abrir no Jupyter
jupyter notebook notebooks/desafio_transformadores.ipynb

# Opção B: Executar via linha de comando (gera outputs e salva imagens)
jupyter nbconvert --to notebook --execute \\
    notebooks/desafio_transformadores.ipynb \\
    --output desafio_transformadores.ipynb \\
    --ExecutePreprocessor.timeout=900
```

### 4. Reconstruir o notebook a partir do script (opcional)

O notebook é gerado de forma idempotente por `scripts/build_notebook.py`:

```bash
python scripts/build_notebook.py
jupyter nbconvert --to notebook --execute notebooks/desafio_transformadores.ipynb --output desafio_transformadores.ipynb
```

## Funções auxiliares (`src/utils.py`)

Módulo único com 4 grupos:

- **Data** — `scale_features`, `make_sliding_windows`.
- **Modelos** — `build_autoencoder`, `build_lstm`, `build_mlp`.
- **Avaliação** — `reconstruction_mse`, `anomaly_threshold`, `classification_report_dict`, `forecast_metrics`.
- **Plots** — `plot_class_distribution`, `plot_correlation`, `plot_time_series_overview`, `plot_training_history`, `plot_anomaly_histogram`, `plot_anomaly_timeline`, `plot_forecast_vs_real`, `plot_confusion_matrix`, `plot_ot_behavior`, `plot_ot_rolling`.
- **Utilitário de relatório** — `save_fig(fig, name)` salva PNGs em `reports/images/` (cria a pasta se preciso).

## Artefatos visuais

Após executar o notebook, **13 figuras** são salvas em `reports/images/`:

| # | Arquivo | Conteúdo |
|---|---|---|
| 01 | `01_time_series_overview.png` | Painel 4×2 das 7 features ao longo do tempo |
| 02 | `02_class_distribution.png` | Barras com a distribuição de `class` |
| 03 | `03_correlation_heatmap.png` | Matriz de correlação entre features |
| 04 | `04_autoencoder_training_history.png` | Curvas de loss do AE |
| 05 | `05_lstm_training_history.png` | Curvas de loss da LSTM |
| 06 | `06_mlp_training_history.png` | Curvas de loss do MLP |
| 07 | `07_ae_anomaly_histogram.png` | Distribuição do erro + threshold p99 |
| 08 | `08_ae_anomaly_timeline.png` | Série temporal com anomalias destacadas |
| 09 | `09_ae_confusion_matrix.png` | AE como detector de falha vs `class` real |
| 10 | `10_lstm_forecast_vs_real.png` | Real × previsto para os 4 horizontes |
| 11 | `11_mlp_confusion_matrix.png` | Confusão da classificação binária |
| 12 | `12_ot_behavior_analysis.png` | Análise focada em OT: distribuição, sazonalidade diária/anual, normal vs falha |
| 13 | `13_ot_rolling.png` | Evolução temporal de OT com média e desvio móveis (janela semanal) |

## Análise da temperatura do óleo (OT)

O notebook inclui uma seção dedicada (2.7) ao comportamento da variável-alvo
do problema:

- **Distribuição global de OT:** média 13.3 ºC, desvio 8.6, amplitude [-4.1, 46.0].
- **Sazonalidade diária e anual** visíveis nos boxplots por hora e por mês.
- **Achado importante:** `OT|class=1` (falha) tem média **6.3 ºC**, contra
  **13.4 ºC** em `class=0` — falhas tendem a ocorrer em **temperaturas mais
  baixas**, sugerindo associação com estados de carga reduzida ou
  desligamento, e não com sobreaquecimento.
- **Janela móvel semanal** revela deriva ao longo dos 2 anos, parcialmente
  responsável pela mudança de distribuição entre treino e teste.

## Dataset

| Coluna | Descrição |
|---|---|
| `date` | Data/hora da medição (granularidade horária) |
| `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL` | Cargas elétricas em diferentes regiões |
| `OT` | Temperatura do óleo do transformador (alvo da LSTM) |
| `class` | Indicador binário (0 = normal, 1 = falha) |

- Período: **2016-07-03 a 2018-06-25**
- Total de registros: **17.348**
- Falhas: **173** (~1.0 %)
- Sem valores faltantes ou duplicatas

## Referências

- **Aula 8 — Autoencoders** (slides do curso, em `docs/referencias/`).
- **Nielsen, A.** *Análise Prática de Séries Temporais — Predição com Estatística e Aprendizado de Máquina.* O'Reilly.
- **Auffarth, B.** *Machine Learning for Time-Series with Python.* Packt Publishing, 2021.
- Hawkins, D. M. *Identification of Outliers.* Chapman & Hall, 1980.

Resumos analíticos das três referências estão em `docs/resumos/`.

## Licença

Projeto acadêmico — uso educacional.
"""

PATH = Path(__file__).resolve().parents[1] / "README.md"
PATH.write_text(CONTENT, encoding="utf-8")
print(f"README.md reescrito em UTF-8 ({PATH.stat().st_size} bytes)")

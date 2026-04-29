# Técnicas Extraídas do Notebook de Referência

> **Fonte:** [docs/referencias/Colab_Autoencoders.ipynb](../referencias/Colab_Autoencoders.ipynb)
>
> Este documento sintetiza as técnicas utilizadas no notebook de aula
> (`Colab_Autoencoders.ipynb`) e serve de **base de referência** para a
> construção do notebook do desafio dos transformadores
> (`data/transformers.csv`). As escolhas (frameworks, padrões de código,
> hiperparâmetros) são listadas de forma direta para reuso.

---

## 1. Stack tecnológico

| Categoria              | Bibliotecas                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| Manipulação numérica   | `numpy`                                                            |
| Deep Learning          | `tensorflow` / `tensorflow.keras` (`layers`, `Model`, `Sequential`)|
| Pré-processamento      | `sklearn.preprocessing.StandardScaler`                             |
| Split                  | `sklearn.model_selection.train_test_split`                         |
| Redução de dimensão    | `sklearn.decomposition.PCA`, `sklearn.manifold.TSNE`               |
| Clusterização          | `sklearn.cluster.KMeans`                                           |
| Visualização           | `matplotlib.pyplot`                                                |

> **Observação:** o notebook usa **TensorFlow/Keras** (não PyTorch). O
> framework do desafio deve seguir o mesmo stack para coerência didática.

---

## 2. Estrutura geral em 5 blocos

O notebook segue um pipeline canônico e modular, repetido a cada caso de uso:

1. **Setup** — `np.random.seed(42)` para reprodutibilidade.
2. **Dados** — geração ou carga + `train_test_split` + `StandardScaler`.
3. **Modelo** — `keras.Input` → `Dense` → `Model` (Functional API ou
   `Sequential`).
4. **Treino** — `model.compile(optimizer="adam", loss="mse")` + `model.fit(X, X, ...)`.
5. **Análise** — `predict`, cálculo de métricas, plots comparativos.

---

## 3. Bloco 1 — Pré-processamento e EDA (cell 0)

### Técnicas

- **Reprodutibilidade:** `np.random.seed(42)`.
- **Geração de correlação artificial entre features** (útil para validar que o
  AE aprende relações lineares simples):
  ```python
  X[:, 5] = 0.3 * X[:, 0] + 0.7 * X[:, 2] + np.random.normal(0, 0.1, len(X))
  ```
- **Split:** `train_test_split(X, test_size=0.2, random_state=42)`.
- **Normalização:** `StandardScaler` — `fit_transform` no treino,
  `transform` no teste (evita data leakage).
- **`input_dim = X_train.shape[1]`** capturado para parametrizar a rede.
- **Visualização padrão (subplots 1×2):**
  - Scatter para checar correlação entre features.
  - Histograma para verificar distribuição pós-normalização (média ~0,
    desvio ~1).

### Aplicação ao desafio
Para `transformers.csv`, manter `random_state=42`, usar `StandardScaler`,
descartar a coluna `class` da entrada do AE, e converter `date` para índice
temporal (não como feature numérica).

---

## 4. Bloco 2 — Autoencoder denso para compressão (cell 1)

### Arquitetura (Functional API)

```
Input(10) → Dense(16, relu) → Dense(8, relu) → Dense(3, linear)  # encoder
Input(3)  → Dense(8, relu)  → Dense(16, relu) → Dense(10, linear) # decoder
autoencoder = Model(encoder_inputs, decoder(encoder(encoder_inputs)))
```

### Padrões a reusar

- **Espaço latente sem ativação** (`Dense(latent_dim, name="latent")`) —
  preserva valores contínuos.
- **Saída com ativação `linear`** — apropriada para dados normalizados sem
  range fixo (caso do `transformers.csv`).
- **Encoder e decoder simétricos** (espelham-se em torno do bottleneck).
- **Encoder e decoder expostos como `keras.Model`** independentes — permite
  `encoder.predict(X)` e `decoder.predict(Z)` de forma isolada.
- **Compilação:** `optimizer="adam"`, `loss="mse"`.
- **Treino auto-supervisionado:** `fit(X_train, X_train, ...)`.

### Hiperparâmetros do notebook

| Parâmetro          | Valor       |
| ------------------ | ----------- |
| `epochs`           | 50          |
| `batch_size`       | 64          |
| `validation_split` | 0.1         |
| `latent_dim`       | 3           |

### Métrica

```python
print("MSE reconstrução:", np.mean((X_test - X_rec) ** 2))
```

---

## 5. Bloco 3 — Visualização de reconstrução (cell 2)

Padrão de plot recomendado para a tarefa de reconstrução: **scatter
sobreposto original vs reconstruído**, uma feature por subplot:

```python
plt.scatter(range(len(X_test)),  X_test[:, i], color='royalblue', alpha=0.5, s=10, label='Original')
plt.scatter(range(len(X_rec)),   X_rec[:, i],  color='darkorange', alpha=0.5, s=10, label='Reconstruída')
```

Detalhes de estilo: `alpha=0.5`, `s=10`, `grid(True, alpha=0.3)`,
`tight_layout()`.

---

## 6. Bloco 4 — Espaço latente + Clusterização + t-SNE (cell 3)

Pipeline para inspecionar o que o AE aprendeu:

1. `Z = encoder.predict(X_train)` → vetores 3D.
2. `KMeans(n_clusters=3, random_state=42).fit(Z)` → labels e `inertia_`.
3. `TSNE(n_components=2, random_state=42).fit_transform(Z)` → projeção 2D.
4. Scatter `Z2[:,0] vs Z2[:,1]` colorido por `kmeans.labels_`, `cmap="tab10"`.

### Aplicação ao desafio
Útil como análise exploratória: verificar se as amostras com `class=1`
formam cluster separado no espaço latente do AE.

---

## 7. Bloco 5 — Detecção de anomalias (cell 4) — **CORE DO DESAFIO**

Este é o bloco mais relevante para o desafio dos transformadores.

### 7.1 Arquitetura (API Sequential, mais enxuta)

```python
autoencoder = models.Sequential([
    layers.Input(shape=(8,)),
    layers.Dense(4, activation='relu'),
    layers.Dense(2, activation='relu'),    # bottleneck
    layers.Dense(4, activation='relu'),
    layers.Dense(8, activation='linear'),  # saída = entrada
])
autoencoder.compile(optimizer='adam', loss='mse')
```

### 7.2 Pipeline de detecção (5 passos)

1. **Treinar somente em dados "normais"**:
   ```python
   autoencoder.fit(X_train, X_train, epochs=30, batch_size=32, verbose=0)
   ```
2. **Reconstruir o conjunto de teste**:
   ```python
   recon = autoencoder.predict(X_test_anom)
   ```
3. **Calcular o erro por amostra** (MSE row-wise):
   ```python
   mse = np.mean(np.square(recon - X_test_anom), axis=1)
   ```
4. **Definir threshold pelo percentil 99**:
   ```python
   thr = np.percentile(mse, 99)
   pred = mse > thr
   ```
5. **Reportar taxa**:
   ```python
   print(f"Taxa de anomalias detectadas: {np.mean(pred)*100:.2f}%")
   ```

### 7.3 Visualização canônica (dois plots)

- **Histograma do erro** com linha vertical no threshold:
  ```python
  plt.hist(mse, bins=50, color='skyblue', edgecolor='black')
  plt.axvline(thr, color='r', linestyle='--', label=f"Threshold = {thr:.3f}")
  ```
- **Série temporal do erro** com pontos vermelhos nas anomalias:
  ```python
  plt.plot(mse, label='Erro de reconstrução', color='blue')
  plt.scatter(np.where(pred)[0], mse[pred], color='red', label='Anomalias detectadas', zorder=5)
  plt.axhline(thr, color='orange', linestyle='--', label='Threshold')
  ```

### 7.4 Truque didático: anomalias artificiais

O notebook injeta ruído gaussiano forte em ~5% das amostras de teste para
demonstrar a detecção:

```python
mask = np.random.rand(len(X_test)) < 0.05
X_test_anom[mask] += np.random.normal(0, 5, X_test.shape[1])
```

> **No desafio dos transformadores**, isso não é necessário (já existe a
> coluna `class` indicando falhas reais). O percentil 99 do erro pode ser
> comparado contra `class` para validar o detector.

---

## 8. Bloco 6 — Variational Autoencoder (cells 5–6)

### 8.1 Componentes-chave

- **Encoder com duas saídas:** `z_mean` e `z_log_var` (sem ativação).
- **Reparametrization trick** via `layers.Lambda`:
  ```python
  def sampling(args):
      z_mean, z_log_var = args
      eps = tf.random.normal(shape=tf.shape(z_mean))
      return z_mean + tf.exp(0.5 * z_log_var) * eps
  z = layers.Lambda(sampling)([z_mean, z_log_var])
  ```
- **Classe customizada `VAE(keras.Model)`** com `train_step` próprio,
  combinando duas perdas:
  - Reconstrução: `tf.reduce_mean(tf.square(data - reconstruction))`
  - KL: `-0.5 * mean(1 + z_log_var - z_mean² - exp(z_log_var))`
- **Decoder com `sigmoid`** (porque o exemplo usa dados em [0, 1]).
- **`tf.GradientTape`** para gradientes manuais quando o modelo tem perda
  customizada.

### 8.2 Visualização do latente

- `PCA(n_components=2)` para projetar originais e reconstruídos em 2D e
  comparar.
- Scatter direto de `z` (já em 2D) para visualizar o espaço latente
  aprendido.

### Aplicação ao desafio
VAE é **opcional/extensão** — útil como segunda abordagem para detecção de
anomalias, complementando o AE denso clássico.

---

## 9. Checklist de técnicas a reusar no notebook do desafio

### Pré-processamento
- [ ] `pd.read_csv("data/transformers.csv")` + `parse_dates=['date']` + set como índice.
- [ ] Separar `X = df.drop(columns=['class'])` e `y = df['class']`.
- [ ] `train_test_split(test_size=0.2, random_state=42)`.
- [ ] `StandardScaler` com `fit_transform` no treino e `transform` no teste.

### Tarefa 1 — Autoencoder para anomalias (replica cell 4)
- [ ] Arquitetura simétrica densa: `7 → 4 → 2 → 4 → 7` (ou similar).
- [ ] `compile(optimizer="adam", loss="mse")`.
- [ ] Treinar **somente em amostras `class==0`** (normais).
- [ ] Calcular `mse` row-wise no conjunto completo.
- [ ] Threshold = `np.percentile(mse, 99)`.
- [ ] Plots: histograma + série temporal com anomalias em vermelho.
- [ ] Comparar `pred` (anomalia detectada) com `class` real → matriz de confusão.

### Análise exploratória opcional (replica cell 3)
- [ ] Encoder → `Z` 2D-3D → `KMeans` + `t-SNE` para visualizar separabilidade
  entre `class=0` e `class=1`.

### Tarefas 2 e 3 (LSTM e Deep MLP)
Estão fora do escopo deste notebook de referência, mas podem reaproveitar:
- O mesmo `StandardScaler` (refit ou pickle).
- Os mesmos padrões de plot (subplots, `alpha`, `grid`).
- O mesmo `random_state=42` para coerência.

---

## 10. Hiperparâmetros sugeridos para o desafio (ponto de partida)

| Parâmetro              | Notebook ref. | Proposta inicial p/ desafio       |
| ---------------------- | ------------- | --------------------------------- |
| `latent_dim`           | 2 ou 3        | 2 ou 3 (7 features → bottleneck)  |
| `epochs`               | 30–50         | 50 (com `EarlyStopping`)          |
| `batch_size`           | 32 ou 64      | 64                                |
| `validation_split`     | 0.1           | 0.1                               |
| `optimizer`            | Adam          | Adam (`learning_rate=1e-3`)       |
| `loss`                 | MSE           | MSE                               |
| `threshold`            | percentil 99  | percentil 99 (conforme enunciado) |
| Ativação interna       | ReLU          | ReLU                              |
| Ativação saída         | linear        | linear                            |

---

## 11. Diferenças importantes entre o notebook de referência e o desafio

| Aspecto              | Notebook ref.                  | Desafio dos transformadores            |
| -------------------- | ------------------------------ | -------------------------------------- |
| Origem dos dados     | Sintéticos (`np.random`)       | Reais (`transformers.csv`, 2 anos)     |
| Rótulos              | Ausentes (puro não-superv.)    | Presentes (`class`) — permite validar  |
| Tipo de série        | Tabular sem tempo              | **Série temporal** (medição horária)   |
| Nº de features       | 8 ou 10 (sintético)            | 7 numéricas (`HUFL,...,LULL,OT`)       |
| Anomalias            | Injetadas artificialmente      | Naturais (eventos reais de falha)      |
| Validação            | Visual (plots)                 | Visual + métricas vs `class` real      |

> **Conclusão:** o notebook de referência fornece o **esqueleto** (split,
> scaler, AE denso, threshold pelo percentil 99, plots de erro). O desafio
> adiciona **dados reais com rótulos**, o que permite ir além e medir
> precisão/recall do detector contra as falhas reais — algo que o notebook
> de referência não faz.

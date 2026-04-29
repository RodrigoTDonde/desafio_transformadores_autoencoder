"""Funções auxiliares para o desafio dos transformadores.

Organizadas em quatro grupos:
    1. Data        — carga, split temporal, scaling, janelas deslizantes
    2. Modelos     — Autoencoder denso, LSTM multivariada, Deep MLP
    3. Avaliação   — MSE de reconstrução, threshold, métricas de classificação/forecast
    4. Plots       — visualizações reutilizáveis (matplotlib/seaborn)
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    recall_score,
)
from sklearn.preprocessing import StandardScaler
from tensorflow import keras
from tensorflow.keras import layers


# =============================================================================
# Diretório padrão para salvar figuras (resolvido a partir da raiz do projeto)
# =============================================================================
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_IMAGES_DIR = _PROJECT_ROOT / "reports" / "images"


def save_fig(
    fig: plt.Figure,
    name: str,
    reports_dir: Union[str, Path] = REPORTS_IMAGES_DIR,
    dpi: int = 150,
) -> Path:
    """Salva a figura em ``reports/images/<name>.png`` (cria a pasta se preciso).

    Returns
    -------
    Path
        Caminho absoluto do arquivo PNG salvo.
    """
    out_dir = Path(reports_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = name if name.lower().endswith(".png") else f"{name}.png"
    out_path = out_dir / filename
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    return out_path



def scale_features(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Padroniza com StandardScaler (fit apenas no treino, evita leakage)."""
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler


def make_sliding_windows(
    X: np.ndarray,
    y: np.ndarray,
    window: int = 24,
    horizon: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """Gera janelas deslizantes (X[t-window:t], y[t:t+horizon])."""
    Xs, ys = [], []
    for i in range(len(X) - window - horizon + 1):
        Xs.append(X[i : i + window])
        ys.append(y[i + window : i + window + horizon])
    return np.asarray(Xs), np.asarray(ys)


# =============================================================================
# 2. Modelos
# =============================================================================
def build_autoencoder(input_dim: int, latent_dim: int = 2) -> keras.Model:
    """Autoencoder denso simétrico para detecção de anomalias.

    Topologia: input_dim -> 4 -> latent_dim -> 4 -> input_dim (linear).
    """
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = layers.Dense(4, activation="relu")(inputs)
    z = layers.Dense(latent_dim, activation="relu", name="latent")(x)
    x = layers.Dense(4, activation="relu")(z)
    outputs = layers.Dense(input_dim, activation="linear", name="reconstruction")(x)
    model = keras.Model(inputs, outputs, name="autoencoder")
    model.compile(optimizer="adam", loss="mse")
    return model


def build_lstm(window: int, n_features: int, horizon: int = 4) -> keras.Model:
    """LSTM multivariada para previsão de horizon passos da temperatura do óleo."""
    model = keras.Sequential(
        [
            keras.Input(shape=(window, n_features)),
            layers.LSTM(64, return_sequences=False),
            layers.Dropout(0.2),
            layers.Dense(32, activation="relu"),
            layers.Dense(horizon, activation="linear"),
        ],
        name="lstm_forecaster",
    )
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    return model


def build_mlp(input_dim: int, dropout: float = 0.3) -> keras.Model:
    """Deep MLP binário com dropout para classificar normal (0) vs falha (1)."""
    model = keras.Sequential(
        [
            keras.Input(shape=(input_dim,)),
            layers.Dense(64, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(32, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(1, activation="sigmoid"),
        ],
        name="deep_mlp_classifier",
    )
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# 3. Avaliação
# =============================================================================
def reconstruction_mse(model: keras.Model, X: np.ndarray) -> np.ndarray:
    """MSE row-wise (uma amostra por linha)."""
    X_rec = model.predict(X, verbose=0)
    return np.mean(np.square(X - X_rec), axis=1)


def anomaly_threshold(mse: np.ndarray, percentile: float = 99.0) -> float:
    """Threshold pelo percentil informado (default 99)."""
    return float(np.percentile(mse, percentile))


def classification_report_dict(
    y_true: np.ndarray, y_pred: np.ndarray
) -> dict:
    """Retorna acc/prec/rec/F1 num único dicionário."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """MAE/RMSE para a previsão multivariada."""
    return {
        "mae": float(mean_absolute_error(y_true.flatten(), y_pred.flatten())),
        "rmse": float(np.sqrt(mean_squared_error(y_true.flatten(), y_pred.flatten()))),
    }


# =============================================================================
# 4. Plots
# =============================================================================
def plot_class_distribution(y: pd.Series, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 3))
    counts = y.value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color=["royalblue", "crimson"])
    for i, v in enumerate(counts.values):
        ax.text(i, v, f"{v}\n({v/len(y)*100:.2f}%)", ha="center", va="bottom")
    ax.set_title("Distribuição da variável class")
    ax.set_xlabel("class")
    ax.set_ylabel("Contagem")
    return ax


def plot_correlation(df: pd.DataFrame, ax=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 5))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Correlação entre features numéricas")
    return ax


def plot_time_series_overview(df: pd.DataFrame, date_col: str = "date"):
    fig, axes = plt.subplots(4, 2, figsize=(14, 8), sharex=True)
    cols = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL", "OT"]
    for ax, col in zip(axes.flat, cols):
        ax.plot(df[date_col], df[col], lw=0.6)
        ax.set_title(col)
        ax.grid(alpha=0.3)
    axes.flat[-1].axis("off")
    fig.tight_layout()
    return fig


def plot_training_history(history, title: str = "Histórico de treinamento"):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history.history["loss"], label="loss (treino)")
    if "val_loss" in history.history:
        ax.plot(history.history["val_loss"], label="loss (validação)")
    ax.set_title(title)
    ax.set_xlabel("Época")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(alpha=0.3)
    return fig


def plot_anomaly_histogram(mse: np.ndarray, threshold: float):
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(mse, bins=60, color="skyblue", edgecolor="black")
    ax.axvline(
        threshold, color="red", linestyle="--", label=f"Threshold p99 = {threshold:.4f}"
    )
    ax.set_title("Distribuição do erro de reconstrução")
    ax.set_xlabel("MSE")
    ax.set_ylabel("Contagem")
    ax.legend()
    return fig


def plot_anomaly_timeline(
    dates: pd.Series,
    mse: np.ndarray,
    threshold: float,
    pred_anom: np.ndarray,
    real_anom: np.ndarray | None = None,
):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(dates, mse, color="steelblue", lw=0.6, label="Erro de reconstrução")
    ax.axhline(
        threshold, color="orange", linestyle="--", label=f"Threshold = {threshold:.4f}"
    )
    ax.scatter(
        dates[pred_anom],
        mse[pred_anom],
        color="red",
        s=12,
        label="Anomalias detectadas",
        zorder=3,
    )
    if real_anom is not None:
        idx_real = np.where(real_anom == 1)[0]
        ax.scatter(
            dates.iloc[idx_real],
            mse[idx_real],
            facecolors="none",
            edgecolors="black",
            s=40,
            label="Falha real (class=1)",
            zorder=4,
        )
    ax.set_title("Detecção de anomalias ao longo do tempo")
    ax.set_xlabel("Data")
    ax.set_ylabel("MSE")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return fig


def plot_forecast_vs_real(
    y_true: np.ndarray, y_pred: np.ndarray, n_samples: int = 200
):
    """Plot dos primeiros n_samples passos da primeira previsão (h=1)."""
    fig, axes = plt.subplots(y_true.shape[1], 1, figsize=(12, 2.5 * y_true.shape[1]))
    if y_true.shape[1] == 1:
        axes = [axes]
    for h in range(y_true.shape[1]):
        ax = axes[h]
        ax.plot(y_true[:n_samples, h], label="Real", color="royalblue")
        ax.plot(y_pred[:n_samples, h], label="Previsto", color="darkorange", alpha=0.8)
        ax.set_title(f"Previsão — passo h={h+1}")
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def plot_ot_behavior(
    df: pd.DataFrame,
    ot_col: str = "OT",
    date_col: str = "date",
    class_col: str = "class",
):
    """Painel 2x2 com o comportamento da temperatura do óleo (OT).

    Subplots:
        (1) Histograma + KDE de OT
        (2) Boxplot de OT por hora do dia (sazonalidade diária)
        (3) Boxplot de OT por mês (sazonalidade anual)
        (4) Distribuição de OT condicionada a class (normal vs falha)
    """
    aux = df.copy()
    aux["hour"] = pd.to_datetime(aux[date_col]).dt.hour
    aux["month"] = pd.to_datetime(aux[date_col]).dt.month

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))

    sns.histplot(aux[ot_col], bins=60, kde=True, color="steelblue", ax=axes[0, 0])
    axes[0, 0].axvline(
        aux[ot_col].mean(),
        color="red",
        linestyle="--",
        label=f"média = {aux[ot_col].mean():.2f} ºC",
    )
    axes[0, 0].set_title("Distribuição de OT")
    axes[0, 0].set_xlabel("Temperatura do óleo (ºC)")
    axes[0, 0].legend()

    sns.boxplot(
        data=aux, x="hour", y=ot_col, ax=axes[0, 1], color="lightblue", fliersize=2
    )
    axes[0, 1].set_title("OT por hora do dia (sazonalidade diária)")
    axes[0, 1].set_xlabel("Hora")
    axes[0, 1].set_ylabel("OT (ºC)")

    sns.boxplot(
        data=aux, x="month", y=ot_col, ax=axes[1, 0], color="lightgreen", fliersize=2
    )
    axes[1, 0].set_title("OT por mês (sazonalidade anual)")
    axes[1, 0].set_xlabel("Mês")
    axes[1, 0].set_ylabel("OT (ºC)")

    if class_col in aux.columns:
        for cls, color, label in [(0, "royalblue", "Normal"), (1, "crimson", "Falha")]:
            subset = aux.loc[aux[class_col] == cls, ot_col]
            if len(subset) > 0:
                sns.kdeplot(
                    subset,
                    ax=axes[1, 1],
                    color=color,
                    fill=True,
                    alpha=0.4,
                    label=f"{label} (n={len(subset)})",
                )
        axes[1, 1].set_title("OT — Normal vs Falha")
        axes[1, 1].set_xlabel("OT (ºC)")
        axes[1, 1].legend()
    else:
        axes[1, 1].axis("off")

    fig.tight_layout()
    return fig


def plot_ot_rolling(
    df: pd.DataFrame,
    ot_col: str = "OT",
    date_col: str = "date",
    window: int = 24 * 7,
):
    """Série temporal de OT com média e desvio móveis (default: janela semanal)."""
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(df[date_col], df[ot_col], color="lightgray", lw=0.5, label="OT (bruto)")
    rolling_mean = df[ot_col].rolling(window=window, min_periods=1).mean()
    rolling_std = df[ot_col].rolling(window=window, min_periods=1).std()
    ax.plot(
        df[date_col], rolling_mean, color="red", lw=1.2, label=f"média móvel ({window}h)"
    )
    ax.fill_between(
        df[date_col],
        rolling_mean - rolling_std,
        rolling_mean + rolling_std,
        color="red",
        alpha=0.15,
        label="±1 desvio móvel",
    )
    ax.set_title(f"Evolução de OT — média e desvio móveis ({window}h)")
    ax.set_xlabel("Data")
    ax.set_ylabel("OT (ºC)")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, labels=("Normal", "Falha")
):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
    )
    ax.set_xlabel("Previsto")
    ax.set_ylabel("Real")
    ax.set_title("Matriz de confusão")
    return fig

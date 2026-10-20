"""
Entrenamiento y evaluación de modelos de clasificación (Avance 2).
Importa el preprocesamiento de ft_engineering.py para comparar modelos en las mismas condiciones.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from ft_engineering import (
    RANDOM_STATE,
    TARGET_COLUMN,
    build_feature_pipeline,
    categoric_features,
    categoric_ordinal_features,
    fit_transform_datasets,
    get_all_feature_columns,
    load_data,
    numeric_features,
    ordinal_categories,
    split_features_target,
)


def build_model(model, X_train, y_train):
    """Entrena cualquier estimador de sklearn y lo retorna ajustado."""
    model.fit(X_train, y_train)
    return model


def summarize_classification(model, X_test, y_test, model_name: str) -> dict:
    """
    Métricas de clasificación:
    - accuracy: aciertos totales (puede engañar con clases desbalanceadas).
    - precision: de los predichos positivos, cuántos lo eran.
    - recall: de los positivos reales, cuántos detectamos.
    - f1: balance entre precision y recall.
    - roc_auc: capacidad de separar clases en distintos umbrales.
    """
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = y_pred

    return {
        "model_name": model_name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_score),
    }


def train_candidate_models(X_train, y_train) -> dict:
    """Entrena tres modelos base: lineal, bosque aleatorio y gradient boosting."""
    candidates = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        "RandomForest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    return {
        name: build_model(estimator, X_train, y_train)
        for name, estimator in candidates.items()
    }


def build_summary_table(trained_models: dict, X_test, y_test) -> pd.DataFrame:
    """Tabla comparativa: una fila por modelo, columnas por métrica."""
    rows = [
        summarize_classification(model, X_test, y_test, model_name)
        for model_name, model in trained_models.items()
    ]
    return pd.DataFrame(rows).set_index("model_name")


def plot_metric_comparison(summary_df: pd.DataFrame) -> None:
    """Barras comparando F1 y ROC-AUC entre modelos."""
    metrics_to_plot = summary_df[["f1", "roc_auc"]].reset_index()
    melted = metrics_to_plot.melt(
        id_vars="model_name", var_name="metrica", value_name="valor"
    )

    plt.figure(figsize=(8, 5))
    sns.barplot(data=melted, x="model_name", y="valor", hue="metrica")
    plt.title("Comparación de modelos — F1 y ROC-AUC")
    plt.xlabel("Modelo")
    plt.ylabel("Valor de la métrica")
    plt.ylim(0, 1)
    plt.tight_layout()


def plot_roc_curves(trained_models: dict, X_test, y_test) -> None:
    """Curvas ROC superpuestas para ver separación de clases por umbral."""
    plt.figure(figsize=(8, 6))

    for model_name, model in trained_models.items():
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
        else:
            continue

        fpr, tpr, _ = roc_curve(y_test, y_score)
        auc = roc_auc_score(y_test, y_score)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Azar")
    plt.title("Curvas ROC superpuestas")
    plt.xlabel("Tasa de falsos positivos")
    plt.ylabel("Tasa de verdaderos positivos")
    plt.legend()
    plt.tight_layout()


def select_best_model(summary_df: pd.DataFrame, metric: str = "roc_auc") -> str:
    """
    Elige el modelo con mejor ROC-AUC en test.
    En producción también importan consistencia temporal y escalabilidad, no solo la métrica.
    """
    return summary_df[metric].idxmax()


if __name__ == "__main__":
    print("=== Entrenamiento y evaluación ===")

    df = load_data()
    feature_columns = get_all_feature_columns(
        numeric_features, categoric_features, categoric_ordinal_features
    )

    pipeline = build_feature_pipeline(
        numeric_features=numeric_features,
        categoric_features=categoric_features,
        categoric_ordinal_features=categoric_ordinal_features,
        ordinal_categories=ordinal_categories,
    )

    X_train, X_test, y_train, y_test = split_features_target(
        df=df,
        target_column=TARGET_COLUMN,
        feature_columns=feature_columns,
    )

    X_train_transformed, X_test_transformed = fit_transform_datasets(
        pipeline, X_train, X_test
    )

    trained_models = train_candidate_models(X_train_transformed, y_train)
    summary_df = build_summary_table(trained_models, X_test_transformed, y_test)

    print("\n--- Tabla resumen de métricas ---")
    print(summary_df.round(4))

    best_model_name = select_best_model(summary_df, metric="roc_auc")
    print(f"\nMejor modelo según ROC-AUC: {best_model_name}")

    plot_metric_comparison(summary_df)
    plot_roc_curves(trained_models, X_test_transformed, y_test)
    plt.show()

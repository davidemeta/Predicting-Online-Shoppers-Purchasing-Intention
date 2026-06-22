import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

# Metrics
def compute_metrics(model, X_test: np.ndarray, y_test: np.ndarray, model_name: str) -> dict:
    # Predictions
    if hasattr(model, "predict_proba"):
        # scikit-learn model
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    else:
        # Keras model — predict returns probabilities directly
        raw = model.predict(X_test, verbose=0)
        y_proba = raw.ravel()
        y_pred = (y_proba > 0.5).astype(int)

    # Metrics 
    results = {
        "model_name": model_name,
        "y_pred": y_pred,
        "y_proba": y_proba,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "report": classification_report(y_test, y_pred, zero_division=0),
    }
    return results


# Confusion Matrix
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, model_name: str):

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Purchase", "Purchase"],
        yticklabels=["No Purchase", "Purchase"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    plt.show()


# ROC Curve (single model)
def plot_roc_curve(y_true: np.ndarray, y_proba: np.ndarray, model_name: str):

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, lw=2, label=f"{model_name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


# Comparative ROC Curves
def plot_comparative_roc(results_dict: dict, save_path: str | None = None):

    colors = ["#2196F3", "#FF5722", "#4CAF50", "#9C27B0", "#FF9800"]
    fig, ax = plt.subplots(figsize=(8, 7))

    for idx, (name, data) in enumerate(results_dict.items()):
        y_true = data["y_true"]
        y_proba = data["y_proba"]
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, lw=2, color=color, label=f"{name} (AUC = {auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("Comparative ROC Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, format='pdf', bbox_inches='tight')
    plt.show()


# SHAP Analysis
def shap_analysis(
    model,
    X_test: np.ndarray,
    feature_names: list,
    save_dir: str | None = None,
):
    
    import shap
    from pathlib import Path

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For binary classification, TreeExplainer returns values for both classes.
    # We select class 1 (positive / purchase) for interpretability.
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        shap_vals = shap_values[..., 1]
    else:
        shap_vals = shap_values

    # Summary plot (beeswarm)
    print("=" * 60)
    print("SHAP Summary Plot (Beeswarm)")
    print("=" * 60)
    shap.summary_plot(
        shap_vals, X_test, feature_names=feature_names,
        plot_size=(12, 8), show=False
    )
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "shap_beeswarm.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # Bar plot (mean |SHAP|)
    print("=" * 60)
    print("SHAP Feature Importance (Bar Plot)")
    print("=" * 60)
    shap.summary_plot(
        shap_vals, X_test, feature_names=feature_names,
        plot_type="bar", plot_size=(12, 8), show=False
    )
    plt.tight_layout()
    if save_dir:
        plt.savefig(Path(save_dir) / "shap_bar.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    return shap_vals

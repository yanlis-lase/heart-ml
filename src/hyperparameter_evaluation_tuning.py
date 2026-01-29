import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """
    Evaluate a fitted binary classification model on the test set.
    Returns metrics + confusion matrix for logging/reporting.
    """
    y_pred = model.predict(X_test)

    # ROC-AUC requires probability estimates (or decision function)
    if hasattr(model, "predict_proba"):
        y_score = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        y_score = model.decision_function(X_test)
    else:
        y_score = None

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "ROC_AUC": roc_auc_score(y_test, y_score) if y_score is not None else None,
        "ConfusionMatrix": confusion_matrix(y_test, y_pred)
    }
    return metrics


def print_metrics(metrics: dict) -> None:
    print(f"\n=== FINAL TEST RESULT: {metrics['Model']} ===")
    print(f"Accuracy : {metrics['Accuracy']:.6f}")
    print(f"Precision: {metrics['Precision']:.6f}")
    print(f"Recall   : {metrics['Recall']:.6f}")
    print(f"F1-score : {metrics['F1']:.6f}")
    if metrics["ROC_AUC"] is not None:
        print(f"ROC-AUC  : {metrics['ROC_AUC']:.6f}")
    else:
        print("ROC-AUC  : N/A (no score/proba available)")
    print("Confusion Matrix:")
    print(metrics["ConfusionMatrix"])


def main():
    # 1) Load dataset
    df = pd.read_excel("data/heart_clean.xlsx")
    X = df.drop(columns=["target"])
    y = df["target"]

    # 2) Split (HARUS sama dengan baseline)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3) (Optional) Baseline Logistic Regression for comparison (no tuning here)
    logreg_baseline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, random_state=42))
    ])
    logreg_baseline.fit(X_train, y_train)

    # 4) KNN tuning
    knn_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier())
    ])

    knn_param_grid = {
        "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
        "clf__weights": ["uniform", "distance"],
        # Kalau mau tambah metric (opsional), buka komentar ini:
        # "clf__metric": ["euclidean", "manhattan"]
    }

    knn_search = GridSearchCV(
        knn_pipeline,
        knn_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    knn_search.fit(X_train, y_train)

    # 5) Decision Tree tuning
    dt_param_grid = {
        "max_depth": [2, 3, 4, 5, None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }

    dt_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        dt_param_grid,
        cv=5,
        scoring="f1",
        n_jobs=-1
    )
    dt_search.fit(X_train, y_train)

    # 6) Print best params + best CV scores (for report screenshot)
    print("=== GRID SEARCH RESULTS (CV=5, scoring=F1) ===")
    print("Best KNN parameters:", knn_search.best_params_)
    print("Best KNN CV F1-score:", knn_search.best_score_)

    print("Best Decision Tree parameters:", dt_search.best_params_)
    print("Best Decision Tree CV F1-score:", dt_search.best_score_)

    # 7) Final evaluation on TEST set (for report table)
    best_knn = knn_search.best_estimator_
    best_dt = dt_search.best_estimator_

    all_metrics = []
    all_metrics.append(evaluate_model(logreg_baseline, X_test, y_test, "Logistic Regression (Baseline)"))
    all_metrics.append(evaluate_model(best_knn, X_test, y_test, "KNN (Tuned)"))
    all_metrics.append(evaluate_model(best_dt, X_test, y_test, "Decision Tree (Tuned)"))

    for m in all_metrics:
        print_metrics(m)

    # 8) Summary table (optional print for easy copy to report)
    summary_rows = []
    for m in all_metrics:
        summary_rows.append({
            "Model": m["Model"],
            "Accuracy": round(m["Accuracy"], 6),
            "Precision": round(m["Precision"], 6),
            "Recall": round(m["Recall"], 6),
            "F1": round(m["F1"], 6),
            "ROC_AUC": round(m["ROC_AUC"], 6) if m["ROC_AUC"] is not None else None
        })

    print("\n=== SUMMARY (copy to report) ===")
    print(pd.DataFrame(summary_rows))


if __name__ == "__main__":
    main()

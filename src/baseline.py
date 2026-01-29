import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score,
    recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from fit_observation import observe_fit


def main():
    # 1) Load dataset
    df = pd.read_excel("data/heart_clean.xlsx")

    X = df.drop(columns=["target"])
    y = df["target"]

    # 2) Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # 3) Define baseline models
    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "KNN": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", KNeighborsClassifier())
        ]),
        "Decision Tree": DecisionTreeClassifier(random_state=42)
    }

    # 4) Train & evaluate
    results = []

    for name, model in models.items():
        observe_fit(model, X_train, y_train, X_test, y_test, name)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] \
            if hasattr(model, "predict_proba") else None

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
            "ROC_AUC": roc_auc_score(y_test, y_proba)
            if y_proba is not None else None,
            "ConfusionMatrix": confusion_matrix(y_test, y_pred)
        })

    results_df = pd.DataFrame(results)
    print("\n=== BASELINE RESULTS ===")
    print(results_df)


if __name__ == "__main__":
    main()

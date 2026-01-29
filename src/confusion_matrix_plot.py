import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import ConfusionMatrixDisplay


# ===============================
# 1. Load dataset
# ===============================
df = pd.read_excel("data/heart_clean.xlsx")

X = df.drop(columns=["target"])
y = df["target"]

# ===============================
# 2. Train-test split
# ===============================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ===============================
# 3. Baseline Logistic Regression
# ===============================
logreg_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, random_state=42))
])

logreg_pipeline.fit(X_train, y_train)

# ===============================
# 4. Rebuild tuning result (KNN)
# ===============================
knn_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", KNeighborsClassifier())
])

knn_param_grid = {
    "clf__n_neighbors": [3, 5, 7, 9, 11, 15],
    "clf__weights": ["uniform", "distance"]
}

knn_search = GridSearchCV(
    knn_pipeline,
    knn_param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1
)

knn_search.fit(X_train, y_train)
best_knn = knn_search.best_estimator_

# ===============================
# 5. Rebuild tuning result (Decision Tree)
# ===============================
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
best_dt = dt_search.best_estimator_

# ===============================
# 6. Plot confusion matrix
# ===============================
def plot_cm(model, title):
    ConfusionMatrixDisplay.from_estimator(
        model,
        X_test,
        y_test,
        cmap="Blues"
    )
    plt.title(title)
    plt.show()


plot_cm(logreg_pipeline, "Confusion Matrix - Logistic Regression")
plot_cm(best_knn, "Confusion Matrix - KNN Tuned")
plot_cm(best_dt, "Confusion Matrix - Decision Tree Tuned")

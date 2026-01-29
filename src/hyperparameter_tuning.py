import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

# 1) Load dataset
df = pd.read_excel("data/heart_clean.xlsx")
X = df.drop(columns=["target"])
y = df["target"]

# 2) Split (HARUS sama dengan baseline)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3) KNN tuning
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

# 4) Decision Tree tuning
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

# 5) Output
print("Best KNN parameters:", knn_search.best_params_)
print("Best KNN CV F1-score:", knn_search.best_score_)

print("Best Decision Tree parameters:", dt_search.best_params_)
print("Best Decision Tree CV F1-score:", dt_search.best_score_)

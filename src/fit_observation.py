from sklearn.model_selection import cross_val_score
import numpy as np

def observe_fit(model, X_train, y_train, X_test, y_test, name):
    model.fit(X_train, y_train)

    train_score = model.score(X_train, y_train)
    cv_score = cross_val_score(
        model, X_train, y_train,
        cv=5, scoring="accuracy"
    ).mean()
    test_score = model.score(X_test, y_test)

    print(f"\n{name}")
    print(f"Train Accuracy : {train_score:.3f}")
    print(f"CV Accuracy    : {cv_score:.3f}")
    print(f"Test Accuracy  : {test_score:.3f}")

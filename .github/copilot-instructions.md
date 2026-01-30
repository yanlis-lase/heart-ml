# Heart Disease ML Project - AI Coding Guidelines

## Project Overview
This is a machine learning project for heart disease prediction using scikit-learn. The codebase consists of standalone Python scripts in `src/` that process data from `data/heart_clean.xlsx` and output results to `output/`.

## Key Conventions
- **Data Loading**: Always load dataset with `pd.read_excel("data/heart_clean.xlsx")`
- **Train/Test Split**: Use stratified split with `train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)`
- **Model Pipelines**: Wrap classifiers in `Pipeline([("scaler", StandardScaler()), ("clf", Classifier)])` for LogReg and KNN; DecisionTree uses no scaler
- **Evaluation Metrics**: Report Accuracy, Precision, Recall, F1, ROC_AUC, ConfusionMatrix for all models
- **Hyperparameter Tuning**: Use `GridSearchCV` with `cv=5`, `scoring="f1"`, `n_jobs=-1`

## Workflow Patterns
- **Baseline Training**: Use `observe_fit(model, X_train, y_train, X_test, y_test, name)` from `fit_observation.py` for quick train/CV/test accuracy logging
- **Full Evaluation**: Follow pattern in `hyperparameter_evaluation_tuning.py` - tune models, evaluate on test set, print detailed metrics
- **Visualization**: Generate plots with seaborn/matplotlib, save to `output/` with `dpi=300, bbox_inches="tight"`
- **Script Execution**: Run from project root: `python src/script_name.py`

## Model Configurations
- **LogisticRegression**: `max_iter=1000, random_state=42`
- **KNeighborsClassifier**: Tune `n_neighbors` [3,5,7,9,11,15], `weights` ["uniform", "distance"]
- **DecisionTreeClassifier**: Tune `max_depth` [2,3,4,5,None], `min_samples_split` [2,5,10], `min_samples_leaf` [1,2,4], `random_state=42`

## File Structure Reference
- `src/hyperparameter_evaluation_tuning.py`: Complete tuning and evaluation pipeline
- `src/data_visualization.py`: Example of plot generation and saving
- `src/descriptive_stats_table.py`: Data preprocessing and table export pattern
- `output/`: All generated files (plots, tables, CSVs) go here

## Dependencies
Install from `requirements.txt`: pandas, numpy, scikit-learn, openpyxl, matplotlib, seaborn</content>
<parameter name="filePath">d:/heart-ml/.github/copilot-instructions.md
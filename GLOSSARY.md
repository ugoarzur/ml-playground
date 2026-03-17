# Glossary — Machine Learning Terms

Terms and definitions used across this project's codebase and notebooks.

| Term | Meaning |
|------|---------|
| **Feature** | An input variable used by the model to make predictions (the columns in `X`). |
| **Target** | The variable the model tries to predict (`y`). Also called label or dependent variable. |
| **Feature Engineering** | Creating new features from existing data to improve model performance (e.g. `FamilySize`, `IsAlone`, `AgeBin`). |
| **Train/Test Split** | Dividing a dataset into a training set (to learn from) and a test set (to evaluate on). Typical ratio: 75-80% train, 20-25% test. |
| **Overfitting** | When a model memorizes training data instead of learning generalizable patterns — performs well on training data but poorly on unseen data. |
| **MinMaxScaler** | Normalization technique that scales features to a 0-1 range. Required when features have different scales. |
| **Confusion Matrix** | A table comparing predicted vs actual classifications. Shows true positives, true negatives, false positives, and false negatives. |
| **Accuracy Score** | Ratio of correct predictions to total predictions. Used for classification tasks. |
| **R2 Score** | Regression metric measuring how well predictions match actual values. Best possible score is 1.0. |
| **Cross-Validation (CV)** | Technique that splits data into multiple folds for training and validation, reducing evaluation bias. `cv=5` means 5-fold cross-validation. |
| **Hyperparameter Tuning** | Searching for the best model configuration parameters (e.g. `n_neighbors`, `learning_rate`, `max_iter`) rather than learning them from data. |
| **GridSearchCV** | Exhaustive search over a specified parameter grid combined with cross-validation to find optimal hyperparameters. |
| **KNeighborsClassifier (KNN)** | Classification algorithm that predicts based on the `k` closest data points in the feature space. |
| **LinearRegression** | Model fitting a linear relationship between features and target by minimizing the residual sum of squares. |
| **PolynomialFeatures** | Transforms features by generating polynomial combinations (e.g. 8 features become 45), capturing non-linear relationships. |
| **HistGradientBoostingRegressor** | Optimized gradient boosting regression model. Builds trees sequentially, each correcting the previous one's errors. |
| **RandomForestRegressor** | Ensemble model that trains many decision trees on random subsets and averages their predictions. |
| **joblib** | Library used to serialize (save) and deserialize (load) trained models to/from disk (`.pkl`, `.joblib` files). |
| **random_state** | Seed for reproducibility — ensures the same random splits and results across runs. |
| **DataFrame** | Pandas tabular data structure (rows and columns) used throughout for data manipulation and preprocessing. |

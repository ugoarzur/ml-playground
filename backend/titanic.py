import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# train_test_split is for: create a training set and testing set
# GridSearchCV: test multiple scenarios for best performances
# MinMaxScaler: cleaning data for the model (convert to a number from 0 to 1)
# KNeighborsClassifier: the model we're using
# Data vizualization
import seaborn as sns

# Machine Learning
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

# Looking at the info we can see there is 86 missing Age data and 327 Cabin missing
# We need to fill theses data as number from 0 to 1


# Data cleaning
def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop unused columns, fill null values and convert in number type
    """
    df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"], inplace=True)

    # Fill the missing values as "S" for Southampton, the most common embarkation point in the data
    df["Embarked"] = df["Embarked"].fillna("S")
    df.drop(columns=["Embarked"], inplace=True)

    fill_missing_ages(df)

    # Fill missing Fare values with median fare
    if df["Fare"].isnull().any():
        fare_median = df["Fare"].median()
        print(
            f"Filling {df['Fare'].isnull().sum()} missing Fare values with median: {fare_median}"
        )
        df["Fare"] = df["Fare"].fillna(fare_median)

    # Convert Gender for model
    df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

    # Feature engineering
    df["FamilySize"] = df["SibSp"] + df["Parch"]  # parents + children
    df["IsAlone"] = np.where(
        df["FamilySize"] == 0, 1, 0
    )  # where there is no one then insert 1
    df["FareBin"] = pd.qcut(
        df["Fare"], 4, labels=False
    )  # categorization for ticket prices
    df["AgeBin"] = pd.cut(
        df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False
    )  # bins for ranged age of passengers

    # Final check for any remaining NaN values
    remaining_nans = df.isnull().sum().sum()
    if remaining_nans > 0:
        print(f"⚠️  WARNING: {remaining_nans} NaN values still remain:")
        print(df.isnull().sum())
    else:
        print("✅ All NaN values have been handled successfully!")

    print(df)
    with open("assets/titanic/data_preprocessed.csv", "w") as f:
        df.to_csv(f, index=False)

    return df


# Fill in missing ages
# because: the average age of passengers is different for 1rst class to second and third class
# don't want: the average age if the age is missing for everyone in the boat
# want: the average of the class where the person is missing
# example: if a first class passenger is missing an age, i want to get the median of all the 1rst class passengers
def fill_missing_ages(df: pd.DataFrame) -> pd.DataFrame:
    """
    filling missing ages in dataFrame (df)
    """
    age_fill_map = {}

    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()

    # Apply the median onto df if row["Age"] is null otherwize keep the original age
    df["Age"] = df.apply(
        lambda row: age_fill_map[row["Pclass"]]
        if pd.isnull(row["Age"])
        else row["Age"],
        axis=1,
    )
    # df["Age"].fillna(df["Pclass"].map(age_fill_map), inplace=True)
    print(f"Age fill map: {age_fill_map}")

    return df


def preparing_data(data: pd.DataFrame) -> list:
    """
    Preprocessing data frame in order to have a good base
    """
    print("Preprocessing data...")
    preprocessed_data = preprocess_data(data)

    # Create Features / Target Variables (Make Flashcards)
    X = preprocessed_data.drop(columns=["Survived"])
    y = preprocessed_data["Survived"]

    return train_test_split(X, y, test_size=0.25, random_state=42)


# Prediction and Evaluate
def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's prediction system
    """
    prediction = model.predict(X_test)
    accuracy = accuracy_score(y_test, prediction)
    matrix = confusion_matrix(y_test, prediction)
    return accuracy, matrix


# Hyperparameter tuning - KeyNearestNeighbors
def tune_model(X_train, y_train):
    """
    Hyperparameter tuning evaluate what are the best parameters for the model.
    Algorithm is "KeyNearestNeighbors"
    """
    param_grid = {
        "n_neighbors": range(1, 21),
        "metric": ["euclidean", "manhattan", "minkowski"],
        "weights": ["uniform", "distance"],
    }

    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    print(f"Best parameters: {grid_search.best_params_}")
    return grid_search.best_estimator_


def plot_model(matrix):
    """
    Plot the confusion matrix using seaborn heatmap
    """
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        xticklabels=["Survived", "Not Survived"],
        yticklabels=["Not Survived", "Survived"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()


def checking_missing_values(X_train, X_test):
    train_nans = X_train.isnull().sum().sum()
    test_nans = X_test.isnull().sum().sum()

    if train_nans > 0 or test_nans > 0:
        print("🚨 ERROR: Found NaN values before scaling!")
        print(f"  - X_train NaNs: {train_nans}")
        print(f"  - X_test NaNs: {test_nans}")
        if train_nans > 0:
            print("NaN counts in X_train:")
            print(X_train.isnull().sum())
        if test_nans > 0:
            print("NaN counts in X_test:")
            print(X_test.isnull().sum())
        return

    print("✅ No NaN values found before scaling - proceeding...")


def main():
    # 1. display data and count null values
    print("Working on Titanic dataset")
    data = pd.read_csv("assets/titanic/titanic.csv")
    data.info()
    print(data.isnull().sum())

    # 2. Prepare data (preprocessing)
    X_train, X_test, y_train, y_test = preparing_data(data)
    # print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # 3. Final NaN check before scaling
    checking_missing_values(X_train, X_test)

    # 4. ML Preprocessing
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 5. Training the model and evaluation
    best_model = tune_model(X_train, y_train)
    accuracy, matrix = evaluate_model(best_model, X_test, y_test)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Confusion Matrix:\n{matrix}")

    # 6. Data Visualization
    plot_model(matrix=matrix)


if __name__ == "__main__":
    main()

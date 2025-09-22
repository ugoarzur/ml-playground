import pandas as pd
import numpy as np

def debug_data_cleaning():
    print("=== DEBUGGING NaN VALUES ===")
    
    # Load original data
    data = pd.read_csv("titanic.csv")
    print("\n1. Original data info:")
    print(f"Shape: {data.shape}")
    print("\nNaN counts in original data:")
    print(data.isnull().sum())
    
    # Copy for processing
    df = data.copy()
    
    # Step 1: Drop columns
    print("\n2. After dropping columns...")
    df.drop(columns=["PassengerId","Name","Ticket","Cabin"], inplace=True)
    print("NaN counts after dropping columns:")
    print(df.isnull().sum())
    
    # Step 2: Handle Embarked
    print("\n3. After handling Embarked...")
    df["Embarked"].fillna("S", inplace=True)
    df.drop(columns=["Embarked"], inplace=True)
    print("NaN counts after handling Embarked:")
    print(df.isnull().sum())
    
    # Step 3: Handle Ages
    print("\n4. After handling Ages...")
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
    
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)
    print(f"Age fill map: {age_fill_map}")
    print("NaN counts after handling Age:")
    print(df.isnull().sum())
    
    # Step 4: Convert Sex
    print("\n5. After converting Sex...")
    df["Sex"] = df["Sex"].map({'male': 1, 'female': 0})
    print("NaN counts after Sex conversion:")
    print(df.isnull().sum())
    
    # Step 5: Feature engineering
    print("\n6. After feature engineering...")
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    
    print("NaN counts before fare binning:")
    print(df.isnull().sum())
    
    # Check if Fare has NaN values and handle them
    if df["Fare"].isnull().any():
        print("\n*** FARE HAS NaN VALUES - THIS IS THE PROBLEM! ***")
        print("Rows with NaN Fare:")
        print(df[df["Fare"].isnull()])
        
        # Fill missing Fare with median
        fare_median = df["Fare"].median()
        print(f"Filling missing Fare with median: {fare_median}")
        df["Fare"].fillna(fare_median, inplace=True)
    
    # Now try the fare binning
    try:
        df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
        print("FareBin created successfully!")
    except Exception as e:
        print(f"ERROR creating FareBin: {e}")
        print("Fare statistics:")
        print(df["Fare"].describe())
    
    df["AgeBin"] = pd.cut(df["Age"], bins=[0,12,20,40,60, np.inf], labels=False)
    
    print("\nFinal NaN counts:")
    print(df.isnull().sum())
    
    # Check final features
    X = df.drop(columns=["Survived"])
    print(f"\nFinal X shape: {X.shape}")
    print("Final X columns:", X.columns.tolist())
    print("NaN counts in X:")
    print(X.isnull().sum())
    
    if X.isnull().any().any():
        print("\n*** STILL HAS NaN VALUES! ***")
        print("Rows with NaN values:")
        nan_rows = X[X.isnull().any(axis=1)]
        print(nan_rows)
    else:
        print("\n✅ No NaN values found!")

if __name__ == "__main__":
    debug_data_cleaning()
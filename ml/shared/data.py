from sklearn.model_selection import train_test_split


def split_dataset(X, y, test_size=0.2, random_state=432):
    """Split dataset into (X_train, X_test, y_train, y_test)."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

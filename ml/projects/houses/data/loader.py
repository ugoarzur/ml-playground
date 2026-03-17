from sklearn import datasets


def load_dataset():
    """Fetch California housing data as (X, y) numpy arrays."""
    return datasets.fetch_california_housing(return_X_y=True)

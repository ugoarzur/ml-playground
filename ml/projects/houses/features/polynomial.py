from sklearn.preprocessing import PolynomialFeatures


def create_polynomial_features(X, degree=2):
    """Apply PolynomialFeatures and return (X_poly, transformer).

    The transformer is returned so it can be reused on new data for prediction.
    """
    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    return X_poly, poly

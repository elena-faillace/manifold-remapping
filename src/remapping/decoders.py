"""Linear decoders (Wiener filter variants)."""

from sklearn import linear_model


class WienerFilterRegression:
    """Simple linear decoder wrapping scikit-learn regressors.

    Args:
        regularization: ``None`` (OLS), ``'l2'`` (Ridge), or ``'LARS'``.
    """

    def __init__(self, regularization: str | None = None):
        if regularization is None:
            self.model = linear_model.LinearRegression()
        elif regularization == "l2":
            self.model = linear_model.Ridge()
        elif regularization == "LARS":
            self.model = linear_model.Lars()
        else:
            print(f"Unknown regularization '{regularization}', using OLS.")
            self.model = linear_model.LinearRegression()

    def fit(self, X_train, y_train):
        """Fit the decoder.

        Args:
            X_train: (n_samples, n_features) neural data.
            y_train: (n_samples, n_outputs) target variables.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """Predict from neural data.

        Args:
            X_test: (n_samples, n_features).

        Returns:
            y_pred: (n_samples, n_outputs).
        """
        return self.model.predict(X_test)

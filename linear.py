from numpy import zeros, ones, hstack, dot, mean, std, ndarray, array


class ScalingParams:
    def __init__(self, mean, std):
        self.mean = mean.flatten()
        self.std = std.flatten()


class LinearRegression:
    def __init__(self):
        self.coefs_ = None
        self.intercept_ = None

    def fit(self, X: ndarray, y: ndarray,
            alpha: float = 1.,
            itr: int = 1000, normalize: bool = True):
        n_samples = X.shape[0]
        n_features = X.shape[1]
        theta = zeros(n_features + 1)

        if normalize:
            X, X_params = self._transform_std(X)
            y, y_params = self._transform_std(y)

        X = hstack((ones(n_samples).reshape(-1, 1), X))  # prepended with 1's

        for _ in range(0, itr):
            est = dot(X, theta)
            diff = est - y

            gradient = dot(X.transpose(), diff)
            gradient *= alpha / n_samples

            theta -= gradient

        if normalize:
            theta = self._inv_transform_std(theta, X_params, y_params)

        self.coefs_ = theta[1:]
        self.intercept_ = theta[0]

    def mse(self, X: ndarray, y: ndarray) -> float:
        est = self.predict(X)
        diff = est - y

        return (diff ** 2).sum() / y.shape[0]

    def predict(self, X: ndarray) -> ndarray:
        if self.coefs_ is None or self.intercept_ is None:
            raise AssertionError("Weights are not trained")
        est = dot(X, self.coefs_) + self.intercept_
        return est

    def _transform_std(self, item: ndarray) -> tuple[ndarray, ScalingParams]:
        """Transform the data with standard scaling"""
        means = mean(item, axis=0)
        stds = std(item, axis=0)

        scl = (item - means) / stds

        return (scl, ScalingParams(means, stds))

    def _inv_transform_std(self, thetas: ndarray,
                           X_params: ScalingParams,
                           y_params: ScalingParams) -> ndarray:
        """Inverse transform theta values into the original scale"""
        thetas = thetas.ravel()
        theta_0 = thetas[0]
        theta_j = thetas[1:]

        theta_div_std = theta_j / X_params.std

        theta_0 = (theta_0 - dot(theta_div_std, X_params.mean))
        theta_0 = y_params.std * theta_0 + y_params.mean

        theta_j = y_params.std * theta_div_std

        return hstack((theta_0, theta_j))

    def set_weights(self, weights: list | ndarray):
        if isinstance(weights, list):
            weights = array(weights)

        self.intercept_ = weights[0]
        self.coefs_ = weights[1:]

    def get_weights(self) -> ndarray | None:
        return self.weights()

    def weights(self) -> ndarray | None:
        if self.intercept_ is None or self.coefs_ is None:
            return None
        return hstack((self.intercept_, self.coefs_))

    def __str__(self):
        return str(self.weights())

    def __repr__(self):
        return repr(self.weights())

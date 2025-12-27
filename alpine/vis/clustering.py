import sklearn.cluster as skc


class KMeans:
    """
    KMeans clustering algorithm.
    This class wraps the sklearn KMeans implementation and provides a simple interface for fitting and predicting clusters.
    """

    def __init__(self, n_clusters, random_state=None):
        assert n_clusters > 0, "Number of clusters must be greater than 0"
        assert isinstance(n_clusters, int), "Number of clusters must be an integer"
        assert random_state is None or isinstance(
            random_state, int
        ), "Random state must be an integer or None"
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.model = skc.KMeans(
            n_clusters=self.n_clusters, random_state=self.random_state
        )

    def fit(self, X):
        assert X.ndim == 2, "Input data must be 2D (samples x features)"
        self.model.fit(X)
        return self

    def predict(self, X, new_shape=None):
        pred = self.model.predict(X)
        if new_shape is not None:
            pred = pred.reshape(new_shape)
        return pred

    def fit_predict(self, X, new_shape=None):
        preds = self.model.fit_predict(X)
        if new_shape is not None:
            preds = preds.reshape(new_shape)
        return preds

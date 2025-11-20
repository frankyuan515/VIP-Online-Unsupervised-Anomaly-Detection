import numpy as np
from sklearn.ensemble import IsolationForest


class AdaptiveIsolationForest:
    """
    Lightweight streaming-friendly Adaptive Isolation Forest.

    Idea:
    - Maintain an ensemble of IsolationForest models (each trained on recent data).
    - On each update:
        * If ensemble is not full yet -> add new trees.
        * If full -> randomly replace a fraction of trees with models trained on new data.
    - The anomaly score is the average (inverted) score across all trees.

    This is a pragmatic online-ish baseline to compare with the VAE.
    """

    def __init__(
        self,
        n_estimators: int = 50,
        replace_rate: float = 0.1,
        contamination: float = 0.05,
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.replace_rate = replace_rate
        self.contamination = contamination
        self.random_state = random_state

        # We treat each IsolationForest here as a "sub-model" in an ensemble.
        self.models = []  # list[IsolationForest]
        self._rng = np.random.RandomState(random_state)

    # ------------------------------------------------------------------
    # Internal helper: fit one IsolationForest on X
    # ------------------------------------------------------------------
    def _fit_one(self, X: np.ndarray) -> IsolationForest:
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        seed = self._rng.randint(0, 10_000)
        model = IsolationForest(
            n_estimators=1,
            contamination=self.contamination,
            random_state=seed,
        )
        model.fit(X)
        return model

    # ------------------------------------------------------------------
    # Initial fit: fill the ensemble with trees trained on X
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray):
        """
        Initialize the ensemble using the provided batch X.
        After this, you can call update(X_new) for online adaptation.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self.models = []
        # If n_estimators is large, this may be slow; but OK for small tests.
        for _ in range(self.n_estimators):
            self.models.append(self._fit_one(X))
        return self

    # ------------------------------------------------------------------
    # Online update: adapt ensemble to new batch X
    # ------------------------------------------------------------------
    def update(self, X: np.ndarray):
        """
        Update the ensemble given new data X.
        If the ensemble is not full yet, we just add trees.
        Otherwise, we randomly replace a fraction of existing trees.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self.models:
            return self.fit(X)

        # Number of trees to replace
        k = max(1, int(self.replace_rate * self.n_estimators))
        indices = self._rng.choice(len(self.models), size=k, replace=False)

        for idx in indices:
            self.models[idx] = self._fit_one(X)

        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Return anomaly scores. Higher = more anomalous.

        IsolationForest.score_samples by default gives higher = more normal,
        so we invert (negative).
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self.models:
            raise RuntimeError("AdaptiveIsolationForest has no fitted models yet.")

        scores_list = []
        for m in self.models:
            s = m.score_samples(X)  # higher = more normal
            scores_list.append(-s)  # invert so higher = more anomaly

        scores = np.mean(scores_list, axis=0)
        return scores

    def predict(self, X: np.ndarray, threshold: float = 0.6):
        """
        Simple thresholding on scores.
        Returns labels (1 = anomaly, 0 = normal) and scores.
        """
        scores = self.score_samples(X)
        labels = (scores > threshold).astype(int)
        return labels, scores

    def get_shap_model(self):
        """
        Return a single IsolationForest instance suitable for SHAP.
        Here we simply take the most recently trained tree model.
        """
        if not self.models:
            raise RuntimeError("AdaptiveIsolationForest has no fitted models yet.")
        return self.models[-1]

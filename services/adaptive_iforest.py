# services/adaptive_iforest.py
'''
import numpy as np
from sklearn.ensemble import IsolationForest


class AdaptiveIsolationForest:
    """
    A small wrapper around sklearn's IsolationForest that exposes:
    - partial_fit(X): here we simply refit on X (true incremental IF is not in sklearn)
    - score_samples(X): anomaly scores where higher = more anomalous
    - get_shap_model(): underlying tree model for SHAP
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples="auto",
        contamination="auto",
        random_state: int = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False

    def partial_fit(self, X):
        """
        "Online" update: here we just (re)fit on X.
        The IFOnlineTrainer will control how X is chosen (sliding window, per-room).
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # In this simple version we refit from scratch each time.
        # Trainer is responsible for passing a reasonable batch.
        self.model.fit(X)
        self._is_fitted = True
        return self

    def score_samples(self, X):
        """
        Return anomaly scores with higher = more anomalous.
        IsolationForest.score_samples returns higher = more normal,
        so we invert.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self._is_fitted:
            # If not fitted yet, just return zeros
            return np.zeros(X.shape[0], dtype=float)

        # sklearn IsolationForest: higher score_samples = more normal.
        raw = self.model.score_samples(X)  # shape (n_samples,)
        # invert so that higher means more anomalous
        scores = -raw
        return scores

    def get_shap_model(self):
        """
        Return the underlying tree ensemble model for SHAP TreeExplainer.
        """
        return self.model
'''
#only aif
# services/adaptive_iforest.py

# services/adaptive_iforest.py

import numpy as np
from sklearn.ensemble import IsolationForest


class AdaptiveIsolationForest:
    """
    Wrapper around sklearn IsolationForest with a sliding training history.

    - partial_fit(X): append X to an internal buffer of recent samples,
      refit on the buffer.
    - score_samples(X): higher score = more anomalous (we invert sklearn's scores).
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples="auto",
        contamination="auto",
        random_state: int = 42,
        max_history: int = 200,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
        self._is_fitted = False
        self.max_history = max_history
        self.X_hist = None  # sliding window of recent samples

    def partial_fit(self, X):
        """
        "Online" update: keep a sliding history of samples and refit on it.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.X_hist is None:
            self.X_hist = X
        else:
            self.X_hist = np.vstack([self.X_hist, X])
            if self.X_hist.shape[0] > self.max_history:
                self.X_hist = self.X_hist[-self.max_history :, :]

        # Fit on the history
        self.model.fit(self.X_hist)
        self._is_fitted = True
        return self

    def score_samples(self, X):
        """
        Return anomaly scores with higher = more anomalous.

        sklearn IsolationForest.score_samples:
        - higher scores → more normal
        We invert that.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if not self._is_fitted or self.X_hist is None:
            # Not fitted yet -> return zeros
            return np.zeros(X.shape[0], dtype=float)

        raw = self.model.score_samples(X)  # higher = more normal
        scores = -raw                      # higher = more anomalous
        return scores

    def get_shap_model(self):
        """Return underlying tree ensemble for SHAP TreeExplainer."""
        return self.model

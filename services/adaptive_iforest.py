import numpy as np
from sklearn.ensemble import IsolationForest

class AdaptiveIsolationForest:
    """
    Lightweight streaming-friendly Adaptive Isolation Forest
    inspired by BWOAIF (Hannák et al. 2023).

    Features:
    - rolling tree replacement
    - online adaptation
    - ensemble scoring
    """

    def __init__(self, n_estimators=50, replace_rate=0.1, random_state=42):
        self.n_estimators = n_estimators
        self.replace_rate = replace_rate     # % of trees replaced each update
        self.random_state = random_state

        self.models = []  # list of IsolationForest instances
        self.fitted = False

    def partial_fit(self, X):
        """
        Incremental update:
        - replace oldest trees
        - train new ones on new batch
        """
        X = np.array(X)

        # First-time training: create full forest
        if not self.fitted:
            for _ in range(self.n_estimators):
                model = IsolationForest(
                    n_estimators=1,
                    max_samples='auto',
                    contamination='auto',
                    random_state=np.random.randint(0, 999999)
                )
                model.fit(X)
                self.models.append(model)
            self.fitted = True
            return

        # Replace "replace_rate" fraction of trees
        k = max(1, int(self.n_estimators * self.replace_rate))

        # Remove oldest trees
        self.models = self.models[k:]

        # Add new trees trained on new data
        for _ in range(k):
            model = IsolationForest(
                n_estimators=1,
                max_samples='auto',
                contamination='auto',
                random_state=np.random.randint(0, 999999)
            )
            model.fit(X)
            self.models.append(model)

    def score_samples(self, X):
        """
        Average anomaly score across ensemble.
        (IsolationForest returns negative scores → convert to positive scale)
        """
        X = np.array(X)
        scores = []

        for model in self.models:
            s = -model.score_samples(X)  # larger = more anomalous
            scores.append(s)

        return np.mean(scores, axis=0)

    def predict(self, X, threshold=0.6):
        scores = self.score_samples(X)
        labels = (scores > threshold).astype(int)  # 1 = anomaly
        return labels, scores
    
    def get_shap_model(self):
        """
        Return a single IsolationForest instance suitable for SHAP.
        Here we simply take the most recently trained tree model.
        """
        if not self.models:
            raise RuntimeError("AdaptiveIsolationForest has no fitted models yet.")
        return self.models[-1]

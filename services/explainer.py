import shap
import torch
import numpy as np
from .vae_model import AdaptiveVAE
from .preprocessing import StreamPreprocessor


class AnomalyExplainer:
    """SHAP for reconstruction error attribution."""

    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.explainer = shap.DeepExplainer(
            self.model, torch.randn(100, 60)
        )  # Background data

    def explain(self, sequence, resourceid):
        """SHAP on sequence, inverse to original units."""
        shap_values = self.explainer.shap_values(sequence)
        scaler = self.preprocessor.scalers[resourceid]
        original_scale = scaler.inverse_transform(
            shap_values.reshape(1, -1)
        )  # Back to raw
        return original_scale  # e.g., "CO₂ +180 ppm"

# reducer.py — PCA dimensionality reduction for quantum (configurable qubits)
from __future__ import annotations
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib, os

class QuantumReducer:
    """StandardScaler (fit on train only) + PCA to n_components (=n_qubits). Fit once, transform anywhere."""
    def __init__(self, n_components=8):
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components, random_state=42)
        self._fitted = False

    def fit(self, X: np.ndarray):
        Xs = self.scaler.fit_transform(X)
        self.pca.fit(Xs)
        self._fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        assert self._fitted, "Call fit() first on train features only"
        return self.pca.transform(self.scaler.transform(X))

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"scaler": self.scaler, "pca": self.pca, "n_components": self.n_components}, path)

    @classmethod
    def load(cls, path: str):
        d = joblib.load(path)
        obj = cls(n_components=d["n_components"])
        obj.scaler = d["scaler"]
        obj.pca = d["pca"]
        obj._fitted = True
        return obj

    def info(self):
        return {"n_components": self.n_components, "explained_variance_ratio": self.pca.explained_variance_ratio_.tolist() if self._fitted else None}

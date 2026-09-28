# cnn_baseline.py - classical baseline model for benchmarking
from __future__ import annotations
from typing import Tuple, Dict, Any
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LightCurveCNN(nn.Module):
    """
    1D Convolutional Neural Network for light curve classification.
    
    Takes raw 256-length curves as input and learns temporal features
    through convolutional layers.
    """
    
    def __init__(
        self,
        n_classes: int = 5,
        initial_channels: int = 16,
        dropout: float = 0.3,
    ):
        super().__init__()
        
        self.initial_channels = initial_channels
        
        # Conv block 1: detect broad patterns
        self.conv1 = nn.Conv1d(
            in_channels=1,
            out_channels=initial_channels,
            kernel_size=7,
            padding=3,
        )
        self.bn1 = nn.BatchNorm1d(initial_channels)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)  # 256 -> 128
        
        # Conv block 2: detect harmonic features
        self.conv2 = nn.Conv1d(
            in_channels=initial_channels,
            out_channels=initial_channels * 2,
            kernel_size=5,
            padding=2,
        )
        self.bn2 = nn.BatchNorm1d(initial_channels * 2)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)  # 128 -> 64
        
        # Conv block 3: detect fine-grained fluctuation patterns
        self.conv3 = nn.Conv1d(
            in_channels=initial_channels * 2,
            out_channels=initial_channels * 4,
            kernel_size=3,
            padding=1,
        )
        self.bn3 = nn.BatchNorm1d(initial_channels * 4)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)  # 64 -> 32
        
        # After 3 poolings: 256 -> 32
        self.flat_size = initial_channels * 4 * 32
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.flat_size, 128)
        self.bn_fc1 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 64)
        self.bn_fc2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, n_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x: torch.Tensor of shape (batch, 1, 256) - raw light curve
        
        Returns
        -------
        logits: torch.Tensor of shape (batch, n_classes)
        """
        # Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc1(x)
        x = self.bn_fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn_fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x


class LightCurveSVM:
    """
    Classical SVM on engineered light curve features.
    """
    
    def __init__(
        self,
        kernel: str = "rbf",
        C: float = 1.0,
        gamma: str = "scale",
    ):
        from sklearn import svm
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.clf = svm.SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            random_state=42,
        )
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train the SVM on engineered features."""
        self.clf.fit(X, y)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        return self.clf.predict(X)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        return self.clf.predict_proba(X)
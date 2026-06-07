"""
model.py
========

Deep-learning model definitions (PyTorch) and sequence-dataset utilities for the
Appliances Energy Prediction task.

Three architectures are provided so they can be compared:

* :class:`LSTMRegressor`  - stacked LSTM, the canonical sequence model.
* :class:`GRURegressor`   - lighter gated recurrent alternative.
* :class:`CNNLSTMRegressor` - 1-D convolution front-end that extracts local
  temporal patterns before the LSTM, often stronger on noisy sensor data.

All models take a window of shape ``(batch, seq_len, n_features)`` and predict a
single scalar (next-step ``Appliances``, standardised).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
# Sequence windowing
# --------------------------------------------------------------------------- #
def make_sequences(
    features: np.ndarray, target: np.ndarray, seq_len: int = 6
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a 2-D feature matrix into overlapping windows for sequence models.

    For each position ``t`` the model sees the window ``features[t-seq_len+1 : t+1]``
    (i.e. ending *at* and including row ``t``) and predicts ``target[t]``.

    Crucially the window includes row ``t`` itself. This is leakage-free because the
    feature matrix never contains the raw target: every target-derived column is a
    *shifted* past value (``Appliances_lag_1[t] = appliances[t-1]``, rolling stats
    ``shift(1)``-ed, etc.). Aligning the window's final row with ``target[t]`` gives
    the sequence model access to the same information set as the tabular baselines —
    in particular the dominant lag-1 signal. (An earlier version ended the window at
    ``t-1``, which staled the lag-1 feature by one step and crippled accuracy.)

    Parameters
    ----------
    features:
        Array of shape ``(n_samples, n_features)``.
    target:
        Array of shape ``(n_samples,)``.
    seq_len:
        Number of timesteps in each window (6 -> 1 hour of history).

    Returns
    -------
    (X, y)
        ``X`` of shape ``(n_windows, seq_len, n_features)`` and ``y`` of shape
        ``(n_windows,)``.
    """
    X, y = [], []
    for end in range(seq_len, len(features) + 1):
        X.append(features[end - seq_len:end])   # rows end-seq_len .. end-1
        y.append(target[end - 1])               # target aligned with last window row
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


class SequenceDataset(Dataset):
    """Thin :class:`torch.utils.data.Dataset` wrapper around windowed arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(-1)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


# --------------------------------------------------------------------------- #
# Activation helper
# --------------------------------------------------------------------------- #
def _activation(name: str) -> nn.Module:
    """Map an activation name to its module.

    The recurrent gates (LSTM/GRU) use ``tanh``/``sigmoid`` internally and are not
    configurable; this controls the **dense head** activation, where the choice of
    ``ReLU`` vs ``tanh`` is a genuine design decision (see report §5.3).
    """
    acts = {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        "gelu": nn.GELU,
    }
    name = name.lower()
    if name not in acts:
        raise ValueError(f"Unknown activation '{name}'. Choose from {list(acts)}.")
    return acts[name]()


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #
class LSTMRegressor(nn.Module):
    """Stacked LSTM followed by a small MLP head.

    Dropout is applied between LSTM layers and before the output to regularise.
    ``tanh`` is the LSTM's internal gate non-linearity; the dense head activation is
    configurable via ``activation`` (default ``ReLU``).
    """

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = out[:, -1, :]          # last timestep's hidden state
        return self.head(last)


class GRURegressor(nn.Module):
    """Stacked GRU regressor (fewer parameters than the LSTM)."""

    def __init__(
        self,
        n_features: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        return self.head(out[:, -1, :])


class CNNLSTMRegressor(nn.Module):
    """1-D CNN feature extractor feeding an LSTM.

    The convolution slides over the time axis to learn local temporal motifs
    (e.g. ramps/spikes) which are then summarised by the LSTM. This hybrid is often
    more robust on noisy multivariate sensor streams.
    """

    def __init__(
        self,
        n_features: int,
        cnn_channels: int = 32,
        hidden_size: int = 64,
        num_layers: int = 1,
        dropout: float = 0.2,
        activation: str = "relu",
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_features, cnn_channels, kernel_size=3, padding=1),
            _activation(activation),
            nn.Conv1d(cnn_channels, cnn_channels, kernel_size=3, padding=1),
            _activation(activation),
        )
        self.lstm = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features) -> conv expects (batch, features, seq_len)
        c = self.conv(x.transpose(1, 2))
        c = c.transpose(1, 2)         # back to (batch, seq_len, channels)
        out, _ = self.lstm(c)
        return self.head(out[:, -1, :])


MODEL_REGISTRY = {
    "lstm": LSTMRegressor,
    "gru": GRURegressor,
    "cnn_lstm": CNNLSTMRegressor,
}


def build_model(name: str, n_features: int, **kwargs) -> nn.Module:
    """Factory: instantiate a model by name from :data:`MODEL_REGISTRY`."""
    name = name.lower()
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}.")
    return MODEL_REGISTRY[name](n_features=n_features, **kwargs)


def build_optimizer(
    name: str, params, lr: float = 1e-3, weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    """Factory for optimisers, so the choice is configurable and comparable.

    * ``adam``    - adaptive moments; robust default for noisy RNN gradients.
    * ``rmsprop`` - adaptive RMS scaling; the classic recurrent-net optimiser.
    * ``sgd``     - momentum SGD; included as a non-adaptive reference point.
    """
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return torch.optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer '{name}'. Choose from ['adam','rmsprop','sgd'].")

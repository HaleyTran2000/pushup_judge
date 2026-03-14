"""
models.py – Load and hold all inference models + LSTM classification helper.
"""
import numpy as np
import torch
import torch.nn as nn
from scipy.interpolate import interp1d
from ultralytics import YOLO

from realtime.config import (
    PD_MODEL_PATH, PD_CONF,
    POSE_PATH,
    LSTM_CKPT, LSTM_CFG,
    FEATURE_COLS, T_FIXED, F_DIM, THRESHOLD,
)

# ── Device selection ──────────────────────────────────────────────────────────
DEVICE = (
    torch.device("mps")  if torch.backends.mps.is_available()  else
    torch.device("cuda") if torch.cuda.is_available()          else
    torch.device("cpu")
)


# ── BiLSTM model definition ───────────────────────────────────────────────────

class PushupLSTM(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64,
                 n_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True, bidirectional=True,
        )
        d = hidden * 2
        self.norm  = nn.LayerNorm(d)
        self.drop1 = nn.Dropout(dropout)
        self.fc1   = nn.Linear(d, 64)
        self.relu  = nn.ReLU()
        self.drop2 = nn.Dropout(0.2)
        self.fc2   = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h, _) = self.lstm(x)
        h_cat = torch.cat([h[-2], h[-1]], dim=1)
        return self.fc2(
            self.drop2(self.relu(self.fc1(self.drop1(self.norm(h_cat)))))
        ).squeeze(1)


# ── Normalisation statistics ──────────────────────────────────────────────────
FEAT_MEAN = np.array(LSTM_CFG["feat_mean"], dtype=np.float32)
FEAT_STD  = np.array(LSTM_CFG["feat_std"],  dtype=np.float32)


# ── Model loader ─────────────────────────────────────────────────────────────

def load_models() -> tuple:
    """
    Load and return (person_model, pose_model, lstm_model).
    Prints a summary to stdout.
    """
    print(f"Device: {DEVICE}")

    person_model = YOLO(str(PD_MODEL_PATH))
    print(f"Person detector loaded  ({PD_MODEL_PATH.name},  conf≥{PD_CONF})")

    pose_model = YOLO(str(POSE_PATH))
    print(f"Pose estimator  loaded  ({POSE_PATH.name})")

    lstm_model = PushupLSTM(
        n_features=LSTM_CFG["n_features"],
        hidden=LSTM_CFG["hidden"],
        n_layers=LSTM_CFG["n_layers"],
        dropout=LSTM_CFG["dropout"],
    ).to(DEVICE)
    ckpt = torch.load(LSTM_CKPT, map_location=DEVICE, weights_only=True)
    lstm_model.load_state_dict(ckpt["model_state"])
    lstm_model.eval()
    print(f"LSTM classifier loaded  (T={T_FIXED}, F={F_DIM}, thresh={THRESHOLD})")

    return person_model, pose_model, lstm_model


# ── Classification helpers ────────────────────────────────────────────────────

def resample_rep(feat_list: list, t_fixed: int = T_FIXED) -> np.ndarray:
    """Variable-length list of feature dicts → fixed (t_fixed, F) float32."""
    T_orig = len(feat_list)
    mat = np.array(
        [
            [
                f.get(col, 0.0) if not np.isnan(f.get(col, np.nan)) else 0.0
                for col in FEATURE_COLS
            ]
            for f in feat_list
        ],
        dtype=np.float32,
    )
    if T_orig == 1:
        return np.tile(mat, (t_fixed, 1))
    if T_orig == t_fixed:
        return mat
    t_old = np.linspace(0, 1, T_orig)
    t_new = np.linspace(0, 1, t_fixed)
    out   = np.zeros((t_fixed, len(FEATURE_COLS)), dtype=np.float32)
    for j in range(len(FEATURE_COLS)):
        fn = interp1d(t_old, mat[:, j], kind="linear", fill_value="extrapolate")
        out[:, j] = fn(t_new)
    return out


def normalise(x: np.ndarray) -> np.ndarray:
    """(T, F) → z-score normalised (T, F)."""
    return (x - FEAT_MEAN) / FEAT_STD


def classify_rep(lstm_model: PushupLSTM, feat_list: list) -> tuple[str, float]:
    """
    Classify a single push-up rep.

    Parameters
    ----------
    lstm_model : loaded PushupLSTM (already on DEVICE, in eval mode)
    feat_list  : list of per-frame feature dicts for the rep

    Returns
    -------
    (label, confidence)  –  label ∈ {"correct", "wrong"}
                             confidence ∈ [0, 1]  (probability of chosen label)
    """
    if len(feat_list) < 2:
        return "unknown", 0.0
    x  = normalise(resample_rep(feat_list, T_FIXED))
    Xt = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        p_correct = float(torch.sigmoid(lstm_model(Xt)).item())
    p_wrong = 1.0 - p_correct
    if p_correct >= 0.5:
        return "correct", p_correct
    return "wrong", p_wrong

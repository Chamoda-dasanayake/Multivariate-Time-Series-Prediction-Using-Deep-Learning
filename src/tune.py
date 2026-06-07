"""
tune.py
=======

Systematic hyperparameter optimisation for the deep sequence models.

Implements **Random Search** and **Grid Search** over the key hyperparameters
called out in the assessment brief — learning rate, batch size, number of epochs
(via early stopping), number of recurrent layers, neurons per layer (hidden size),
plus dropout, sequence length, optimiser and activation function.

Methodology (leakage-free)
--------------------------
* The chronological **test set (last 20%) is never touched during tuning.**
* Tuning is scored on an inner **validation** split carved from the tail of the
  training partition; scalers and feature selection are fit on the inner-training
  portion only.
* Each trial trains with early stopping and is ranked by validation **RMSE (Wh)**.
* All trials are logged to ``reports/tuning_results.csv``; the winner is written to
  ``reports/best_hyperparameters.json`` and can be fed straight into ``train.py``.

A note on Bayesian optimisation: Optuna/`scikit-optimize` would slot in here as a
drop-in replacement for the sampler; Random Search is used by default as it needs
no extra dependency and is a strong, embarrassingly-parallel baseline that matches
or beats Grid Search for the same budget (Bergstra & Bengio, 2012).

Usage
-----
    python src/tune.py --method random --trials 20 --epochs 25
    python src/tune.py --method grid --epochs 20
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).resolve().parent))

from data_preprocessing import (  # noqa: E402
    TARGET, drop_noise_columns, handle_missing_values, load_data,
    scale_features, temporal_train_test_split, treat_outliers,
)
from evaluate import regression_metrics  # noqa: E402
from feature_engineering import (  # noqa: E402
    build_features, get_feature_columns, rank_feature_importance, select_top_features,
)
from model import SequenceDataset, build_model, build_optimizer, make_sequences  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

# Hyperparameter search space.
SEARCH_SPACE: dict[str, list] = {
    "model": ["gru", "lstm"],
    "seq_len": [6, 12, 18],
    "hidden_size": [32, 64, 96],
    "num_layers": [1, 2],
    "dropout": [0.1, 0.2, 0.3],
    "lr": [1e-3, 5e-4, 3e-4],
    "batch_size": [64, 128, 256],
    "optimizer": ["adam", "rmsprop"],
    "activation": ["relu", "tanh"],
}

# A deliberately small grid (so Grid Search stays tractable on CPU).
GRID_SPACE: dict[str, list] = {
    "model": ["gru"],
    "seq_len": [12],
    "hidden_size": [64, 96],
    "num_layers": [1, 2],
    "dropout": [0.2, 0.3],
    "lr": [1e-3, 5e-4],
    "batch_size": [128],
    "optimizer": ["adam", "rmsprop"],
    "activation": ["relu"],
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def prepare_data() -> dict:
    """Clean, engineer, split (train/test) and carve an inner validation set.

    Returns a dict of scaled numpy arrays for inner-train / validation, with the
    target scaler so RMSE can be reported in Wh. The test set is intentionally not
    returned here — tuning must not see it.
    """
    df = load_data(ROOT / "data" / "raw" / "energy_data_set.csv")
    df = handle_missing_values(drop_noise_columns(df))
    df, _ = treat_outliers(df)
    feats = build_features(df)

    train_df, _test_df = temporal_train_test_split(feats, train_frac=0.8)

    # Inner split: last 15% of the training partition is the validation set.
    n_inner = int(len(train_df) * 0.85)
    inner_train = train_df.iloc[:n_inner]
    inner_val = train_df.iloc[n_inner:]

    # Feature selection + scaling fit on inner-train only (no leakage).
    cols = get_feature_columns(inner_train)
    importance = rank_feature_importance(inner_train, cols)
    selected = select_top_features(importance, top_k=25)

    tr_s, val_s, _fs, target_scaler = scale_features(inner_train, inner_val, selected, target=TARGET)
    return {
        "feat_train": tr_s[selected].values.astype(np.float32),
        "tgt_train": tr_s[TARGET].values.astype(np.float32),
        "feat_val": val_s[selected].values.astype(np.float32),
        "tgt_val": val_s[TARGET].values.astype(np.float32),
        "target_scaler": target_scaler,
        "n_features": len(selected),
    }


def evaluate_config(cfg: dict, data: dict, epochs: int, patience: int, device: str) -> dict:
    """Train one configuration and return its validation metrics (Wh)."""
    seq_len = cfg["seq_len"]
    Xtr, ytr = make_sequences(data["feat_train"], data["tgt_train"], seq_len)
    Xval, yval = make_sequences(data["feat_val"], data["tgt_val"], seq_len)

    train_loader = DataLoader(SequenceDataset(Xtr, ytr), batch_size=cfg["batch_size"], shuffle=True)
    val_loader = DataLoader(SequenceDataset(Xval, yval), batch_size=256)

    model = build_model(
        cfg["model"], n_features=data["n_features"],
        hidden_size=cfg["hidden_size"], num_layers=cfg["num_layers"],
        dropout=cfg["dropout"], activation=cfg["activation"],
    ).to(device)
    optimiser = build_optimizer(cfg["optimizer"], model.parameters(), lr=cfg["lr"])
    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    no_improve = 0
    used_epochs = 0
    for epoch in range(1, epochs + 1):
        used_epochs = epoch
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()

        model.eval()
        vls = []
        with torch.no_grad():
            for xb, yb in val_loader:
                vls.append(criterion(model(xb.to(device)), yb.to(device)).item())
        v = float(np.mean(vls))
        if v < best_val - 1e-5:
            best_val = v
            best_state = {k: x.cpu().clone() for k, x in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    preds = []
    with torch.no_grad():
        for xb, _ in val_loader:
            preds.append(model(xb.to(device)).cpu().numpy())
    ts = data["target_scaler"]
    pred_wh = ts.inverse_transform(np.concatenate(preds).reshape(-1, 1)).ravel()
    true_wh = ts.inverse_transform(yval.reshape(-1, 1)).ravel()
    metrics = regression_metrics(true_wh, pred_wh)
    metrics["epochs_used"] = used_epochs
    return metrics


# Fixed base config for the controlled activation/optimizer ablation.
ABLATION_BASE: dict = {
    "model": "gru", "seq_len": 12, "hidden_size": 64, "num_layers": 2,
    "dropout": 0.3, "lr": 5e-4, "batch_size": 128,
    "optimizer": "adam", "activation": "relu",
}


def run_ablation(data: dict, epochs: int, patience: int, device: str, seed: int) -> pd.DataFrame:
    """Controlled one-factor-at-a-time study of optimiser and activation.

    Everything except the studied factor is held at :data:`ABLATION_BASE`, so the
    effect of each choice is isolated — the rigorous way to justify the selection.
    """
    trials = []
    for opt in ["adam", "rmsprop", "sgd"]:
        trials.append({**ABLATION_BASE, "optimizer": opt, "_factor": f"optimizer={opt}"})
    for act in ["relu", "tanh", "gelu"]:
        trials.append({**ABLATION_BASE, "activation": act, "_factor": f"activation={act}"})

    rows = []
    for i, cfg in enumerate(trials, 1):
        set_seed(seed)  # same init per trial so differences are due to the factor
        factor = cfg.pop("_factor")
        m = evaluate_config(cfg, data, epochs, patience, device)
        rows.append({"factor": factor, **cfg, **{k: round(v, 4) for k, v in m.items()}})
        print(f"[{i}/{len(trials)}] {factor:18s} -> val RMSE {m['RMSE']:.2f} | "
              f"MAE {m['MAE']:.2f} | R2 {m['R2']:.3f}")
    return pd.DataFrame(rows)


def sample_configs(method: str, n_trials: int, seed: int) -> list[dict]:
    """Generate the list of configurations to evaluate."""
    if method == "grid":
        keys = list(GRID_SPACE)
        combos = list(itertools.product(*[GRID_SPACE[k] for k in keys]))
        return [dict(zip(keys, c)) for c in combos]

    # Random search: sample without replacement where possible.
    rng = random.Random(seed)
    seen, configs = set(), []
    attempts = 0
    while len(configs) < n_trials and attempts < n_trials * 50:
        attempts += 1
        cfg = {k: rng.choice(v) for k, v in SEARCH_SPACE.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            configs.append(cfg)
    return configs


def main(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Preparing data (device={device}) ...")
    data = prepare_data()

    if args.method == "ablation":
        print("Controlled activation / optimizer ablation (base config fixed)\n")
        abl = run_ablation(data, args.epochs, args.patience, device, args.seed)
        out = ROOT / "reports" / "ablation_activation_optimizer.csv"
        abl.to_csv(out, index=False)
        print(f"\nAblation results -> {out}")
        return

    configs = sample_configs(args.method, args.trials, args.seed)
    print(f"{args.method.title()} search over {len(configs)} configurations\n")

    rows = []
    for i, cfg in enumerate(configs, 1):
        metrics = evaluate_config(cfg, data, args.epochs, args.patience, device)
        row = {**cfg, **{k: round(v, 4) for k, v in metrics.items()}}
        rows.append(row)
        print(f"[{i:2d}/{len(configs)}] {cfg['model']:4s} seq{cfg['seq_len']:<2} "
              f"h{cfg['hidden_size']:<2} L{cfg['num_layers']} do{cfg['dropout']} "
              f"lr{cfg['lr']:.0e} bs{cfg['batch_size']:<3} {cfg['optimizer']:7s} "
              f"{cfg['activation']:4s} -> val RMSE {metrics['RMSE']:.2f} | R2 {metrics['R2']:.3f}")

    results = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    out_csv = ROOT / "reports" / f"tuning_results_{args.method}.csv"
    results.to_csv(out_csv, index=False)

    best = results.iloc[0].to_dict()
    best_cfg = {k: best[k] for k in SEARCH_SPACE}
    # Cast numpy types to native for clean JSON.
    best_cfg = {k: (int(v) if isinstance(v, (np.integer,)) else
                    float(v) if isinstance(v, (np.floating,)) else v)
                for k, v in best_cfg.items()}
    best_out = {
        "method": args.method,
        "best_config": best_cfg,
        "validation_metrics": {k: float(best[k]) for k in ["MAE", "RMSE", "MAPE", "R2"]},
    }
    with open(ROOT / "reports" / "best_hyperparameters.json", "w") as fh:
        json.dump(best_out, fh, indent=2)

    print("\n=== Best configuration (by validation RMSE) ===")
    print(json.dumps(best_out, indent=2))
    print(f"\nFull results -> {out_csv}")
    print("Re-train the final model with, e.g.:")
    print(f"  python src/train.py --models {best_cfg['model']} --seq-len {best_cfg['seq_len']} "
          f"--hidden-size {best_cfg['hidden_size']} --num-layers {best_cfg['num_layers']} "
          f"--dropout {best_cfg['dropout']} --lr {best_cfg['lr']} "
          f"--batch-size {best_cfg['batch_size']} --optimizer {best_cfg['optimizer']} "
          f"--activation {best_cfg['activation']}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Hyperparameter tuning for deep models.")
    p.add_argument("--method", default="random", choices=["random", "grid", "ablation"])
    p.add_argument("--trials", type=int, default=20, help="Random-search trials.")
    p.add_argument("--epochs", type=int, default=25, help="Max epochs per trial.")
    p.add_argument("--patience", type=int, default=6, help="Early-stopping patience per trial.")
    p.add_argument("--seed", type=int, default=42)
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

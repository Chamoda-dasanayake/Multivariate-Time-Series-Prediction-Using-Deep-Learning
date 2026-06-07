# Appliance Energy Prediction - Multivariate Time-Series Deep Learning

Predicting household **appliance energy consumption (Wh)** from environmental,
temporal and energy sensor readings sampled at 10-minute intervals, using
classical baselines and deep recurrent models (LSTM / GRU / CNN-LSTM) in PyTorch.

This repository is the deliverable for the *Multivariate Time-Series Prediction
Using Deep Learning* assessment (Appliance Energy Prediction Dataset).

---

## 1. Problem

Given ~19,700 timestamped records of indoor/outdoor temperature & humidity,
weather, and lighting energy, forecast the next-step `Appliances` energy use.
Because the data is a time series, all preprocessing, feature engineering and
splitting is **causal and chronological** to avoid look-ahead leakage.

## 2. Repository Structure

```
.
├── data/
│   ├── raw/energy_data_set.csv          # original dataset
│   └── processed/features.csv           # engineered feature matrix 
├── notebooks/
│   └── EDA.ipynb                        # exploratory analysis walkthrough
├── src/
│   ├── data_preprocessing.py            # load, clean, outliers, scale, split
│   ├── feature_engineering.py           # time/rolling/lag/interaction + selection
│   ├── model.py                         # PyTorch LSTM / GRU / CNN-LSTM + windowing
│   ├── evaluate.py                      # metrics (MAE/RMSE/MAPE/R²) + plots
│   ├── eda.py                           # generates EDA figures
│   ├── tune.py                          # hyperparameter search (random/grid) + ablation
│   └── train.py                         # end-to-end training/evaluation pipeline
├── models/
│   ├── trained_model.pt                 # best deep model weights + config
│   ├── feature_scaler.joblib            # fitted StandardScaler for features
│   ├── target_scaler.joblib             # fitted StandardScaler for target
│   └── selected_features.joblib         # top-k feature names
├── reports/
│   ├── metrics.csv / metrics.json       # final metrics per model
│   ├── feature_importance.csv           # RF importance ranking
│   ├── tuning_results_random.csv        # random search log
│   ├── tuning_results_grid.csv          # grid search log
│   ├── best_hyperparameters.json        # best config from tuning
│   ├── ablation_activation_optimizer.csv  # activation/optimizer study
│   └── figures/                         # all plots (EDA, loss curves, predictions)
├── requirements.txt
└── README.md
```

## 3. Setup

```bash
# 1. Create an isolated environment (conda or venv)
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt
```

Python 3.9+ is recommended. The deep-learning code runs on CPU out of the box and
will automatically use CUDA if a GPU is available.

## 4. How to Run

All commands are run from the repository root.

```bash
# (a) Generate exploratory-data-analysis figures -> reports/figures/
python src/eda.py

# (b) Run the full pipeline: clean -> engineer -> select ->
#     baselines (LinearRegression, RandomForest) -> deep model(s) -> evaluate
python src/train.py --models lstm gru cnn_lstm --epochs 60 --seq-len 12 --top-k 25
```

```bash
# (c) Hyperparameter optimisation (Random / Grid search) on a validation split
python src/tune.py --method random --trials 20 --epochs 25
python src/tune.py --method grid   --epochs 20

# (d) Controlled activation/optimizer study (one factor at a time)
python src/tune.py --method ablation --epochs 25
```

Choose the deep architectures with `--models lstm gru cnn_lstm` (any subset). Useful
`train.py` flags:

| Flag | Default | Meaning |
|------|---------|---------|
| `--models` | `lstm gru cnn_lstm` | Deep architectures to train & compare |
| `--epochs` | `60` | Max training epochs (early stopping may stop sooner) |
| `--seq-len` | `12` | History window length (12 steps = 2 hours) |
| `--top-k` | `25` | Number of features kept after RF-importance selection |
| `--hidden-size` | `64` | Recurrent hidden units (neurons per layer) |
| `--num-layers` | `2` | Number of recurrent layers |
| `--dropout` | `0.3` | Dropout rate (regularisation) |
| `--activation` | `relu` | Dense-head activation: `relu`/`tanh`/`leaky_relu`/`gelu` |
| `--optimizer` | `adam` | Optimiser: `adam`/`rmsprop`/`sgd` |
| `--lr` | `5e-4` | Learning rate |
| `--weight-decay` | `1e-4` | L2 weight decay (regularisation) |
| `--patience` | `12` | Early-stopping patience |

`tune.py` writes `reports/tuning_results_*.csv`, `reports/best_hyperparameters.json`
and `reports/ablation_activation_optimizer.csv`; feed the best config back into
`train.py` via the flags above.

**Outputs after a run:**

* `models/trained_model.pt` — model weights + config
* `models/feature_scaler.joblib`, `target_scaler.joblib`, `selected_features.joblib`
* `reports/metrics.csv` / `metrics.json` — MAE, RMSE, MAPE, R² per model
* `reports/figures/*.png` — loss curves, predicted-vs-actual, residuals, comparison

## 5. Approach Summary

1. **Preprocessing** - time-aware interpolation for missing values, IQR
   winsorising of sensor outliers (target left intact), standardisation fit on
   train only, chronological 80/20 split. Random noise columns `rv1`/`rv2` dropped.
2. **Feature engineering** - calendar/cyclical time features (incl. `NSM`,
   `WeekStatus`, `Day_of_week`), 1h/3h rolling mean & std, autocorrelation-guided
   lags, temperature×humidity interactions; top-k selected by Random-Forest
   importance.
3. **Models** - Linear Regression & (regularised) Random Forest baselines; LSTM,
   GRU and CNN-LSTM deep models with dropout, Adam/RMSprop, MSE loss and early stopping.
4. **Evaluation** - MAE / RMSE / MAPE / R² on the held-out future, with
   predicted-vs-actual and residual diagnostics.
5. **Optimisation** - Random Search (20 trials), Grid Search (32 configs), and
   controlled activation/optimizer ablation study. Best model re-trained with
   optimal hyperparameters.

## 6. Notes

* The brief lists `T1–T6`/`RH_1–RH_6`; the shipped CSV is the full UCI dataset
  (`T1–T9`, `RH_1–RH_9`, plus weather). The code adapts to whatever columns are
  present and reconstructs `NSM`/`WeekStatus`/`Day_of_week` from the timestamp.
* Results are reproducible via the fixed `--seed` (default 42).
* The deep-learning framework used is **PyTorch** - trained models are saved as
  `.pt` files (the PyTorch standard format).

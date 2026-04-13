# FaultSense — Industrial Equipment Fault Predictor

> Real-time binary fault detection for industrial equipment using LightGBM, served via a Flask web application.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![LightGBM](https://img.shields.io/badge/Model-LightGBM-brightgreen)
![Flask](https://img.shields.io/badge/API-Flask-lightgrey)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## Overview

FaultSense takes live sensor readings from industrial equipment — temperature, pressure, vibration, and humidity — and predicts in real time whether the equipment is **healthy** or **faulty**. It includes a full ML pipeline from synthetic data generation through hyperparameter search to a production-ready web interface.

**Equipment types supported:** Pump · Compressor · Motor · Valve · Sensor

---

## Screenshots

The web UI lets you drag sensor sliders and get an instant fault prediction with probability score and confidence level.

> Run the app (see below) and open `http://localhost:5000`

---

## Project Structure

```
FaultSense/
│
├── app.py                          # Flask web app (main entry point)
├── app2.py                         # Alternative app variant
│
├── data_synthesier.py              # Synthetic dataset generator
├── dataset.py                      # Dataset structuring utilities
├── distribution_function.py        # Sensor feature distribution modelling
├── data_analyze.py                 # Exploratory data analysis
├── data.ipynb                      # EDA notebook
│
├── main.py → main8.py              # Iterative experiment scripts
├── main9_by_claude.py              # Claude-assisted experiment
├── main10_claude_combnation.py     # Dense hyperparameter grid search (~13,650 runs)
├── main11.py                       # Final experiment iteration
│
├── synthetic_nim_parallel_10000.csv  # Primary training dataset (10,000 samples)
├── RANDOM_FOREST.csv               # Random Forest baseline results
├── faultsense_model.joblib         # Serialised trained pipeline
│
├── results/                        # Experiment results (CSV / XLSX)
├── plots/                          # Saved diagnostic plots
├── analysis/                       # Additional analysis outputs
├── industrial-equipment-monitoring-dataset/  # Raw dataset folder
└── synthetics3/                    # Additional synthetic data variants
```

---

## Features

- **Binary fault classification** — predicts `FAULTY` or `HEALTHY` with probability score
- **Confidence levels** — HIGH / MEDIUM / LOW based on prediction probability
- **Live web UI** — interactive sliders for all sensor inputs, dark-mode interface
- **Prediction history** — last 20 predictions shown in-session
- **Model info panel** — displays test AUC, F1, accuracy, precision, and recall live in the UI
- **REST API** — `/predict` endpoint accepts JSON for programmatic use
- **Auto train or load** — automatically retrains if no saved model is found

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/techavenger123/Trial_AI_ProjDATASET.git
cd Trial_AI_ProjDATASET
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
python app.py
```

On first run, if `faultsense_model.joblib` is not present, the model will train automatically using `synthetic_nim_parallel_10000.csv`. This takes under a minute.

### 4. Open in browser

```
http://localhost:5000
```

---

## Requirements

```
flask
lightgbm
scikit-learn
pandas
numpy
joblib
matplotlib
tqdm
openpyxl
```

Install all at once:

```bash
pip install flask lightgbm scikit-learn pandas numpy joblib matplotlib tqdm openpyxl
```

Python 3.9 or higher is recommended.

---

## Dataset

The primary dataset (`synthetic_nim_parallel_10000.csv`) contains **10,000 synthetic sensor readings** generated in parallel to simulate realistic industrial conditions.

| Feature | Type | Range | Description |
|---|---|---|---|
| `equipment` | Categorical | pump, compressor, motor, valve, sensor | Equipment type |
| `temperature` | Float | –20 to 120 °C | Operating temperature |
| `pressure` | Float | 0 to 20 bar | Internal pressure |
| `vibration` | Float | 0 to 50 mm/s | Vibration level |
| `humidity` | Float | 0 to 100 % | Ambient humidity |
| `location` | Categorical | — | Installation location (dropped at training) |
| `faulty` | Binary | 0 / 1 | **Target** — 0 = healthy, 1 = faulty |

Class imbalance is handled via `class_weight="balanced"` in the LightGBM classifier.

---

## Model

FaultSense uses a **scikit-learn Pipeline** combining preprocessing and a LightGBM classifier.

### Architecture

```
Input features
    │
    ├── equipment (categorical) ──► OneHotEncoder
    └── temperature, pressure,
        vibration, humidity (numeric) ──► passthrough
                        │
                        ▼
                LGBMClassifier
                        │
                        ▼
              Fault probability [0–1]
                        │
                  threshold = 0.5
                        │
              FAULTY (1) / HEALTHY (0)
```

### Best configuration

| Parameter | Value |
|---|---|
| Learning rate | 0.05 |
| n_estimators | 165 |
| max_depth | 8 |
| num_leaves | 50 |
| subsample | 0.8 |
| colsample_bytree | 0.8 |
| Train / Val / Test split | 90% / 5% / 5% |
| Prediction threshold | 0.5 |

This configuration was selected from a dense grid search of **~13,650 combinations** across 35 learning rates, 78 estimator counts, and 5 train/val/test split ratios (see `main10_claude_combnation.py`).

---

## API Reference

### `POST /predict`

Predict fault status from sensor readings.

**Request body (JSON)**

```json
{
  "equipment": "pump",
  "temperature": 75.5,
  "pressure": 12.3,
  "vibration": 18.0,
  "humidity": 65
}
```

**Response**

```json
{
  "prediction": 1,
  "probability": 0.8732,
  "confidence": "HIGH",
  "threshold": 0.5,
  "label": "FAULTY"
}
```

### `GET /model_info`

Returns the current model configuration and test-set performance metrics.

**Response**

```json
{
  "config": {
    "learning_rate": 0.05,
    "n_estimators": 165,
    "train_ratio": 0.9,
    "val_ratio": 0.05,
    "test_ratio": 0.05
  },
  "test_metrics": {
    "test_auc": 0.97,
    "test_accuracy": 0.94,
    "test_f1": 0.93,
    "test_precision": 0.91,
    "test_recall": 0.95,
    "test_logloss": 0.18
  }
}
```

---

## Running the Hyperparameter Search

To reproduce the full grid search (warning: this takes significant time — ~13,650 model fits):

```bash
python main10_claude_combnation.py
```

Results are saved to `results/synthetic/dense_results.csv` and `dense_results.xlsx`. Six diagnostic plots are saved to `Synthetic1/synthetic_plot/`:

- Validation metric heatmaps (LR × n_estimators)
- Metrics vs n_estimators per split ratio
- Metrics vs learning rate per split ratio
- Train vs validation curves (best split)
- Overfitting heatmap (train AUC − val AUC)
- Top-30 config scatter (val F1 vs val AUC)

The search supports **checkpointing** — if interrupted, it resumes from where it left off.

---

## Retrain from Scratch

To force a full retrain (ignoring any saved model):

```bash
# Delete the saved model, then run the app
rm faultsense_model.joblib
python app.py
```

Or edit `BEST_CONFIG` in `app.py` to change hyperparameters before retraining.

---

## Known Limitations

- **Synthetic data only** — the model has not been validated on real industrial sensor readings. Performance may differ on real-world data.
- **Fixed threshold** — the prediction threshold is set to 0.5. For safety-critical applications, consider tuning this using a precision-recall curve to favour recall (catching more faults at the cost of more false alarms).
- **No feature explainability** — the app does not currently show which sensor reading drove a given prediction. Adding SHAP values would improve interpretability for maintenance engineers.
- **No authentication** — the Flask app runs without any access control. Do not expose it publicly without adding authentication.
- **Single model** — only LightGBM is deployed. Ensemble approaches or periodic retraining on fresh data may improve production reliability.

---

## Development History

This project was built iteratively, with experiment scripts versioned as `main.py` through `main11.py`. Scripts `main9_by_claude.py` and `main10_claude_combnation.py` reflect AI-assisted development using Claude.

---

## License

MIT License. See `LICENSE` for details.

---

## Contributing

Pull requests are welcome. For significant changes, please open an issue first to discuss what you would like to change.
"""
╔══════════════════════════════════════════════════════════╗
║   FaultSense — Random Forest Fault Prediction App       ║
║   Usage: python fault_predictor_app.py                  ║
╚══════════════════════════════════════════════════════════╝

Dependencies:
    pip install flask scikit-learn pandas numpy

How to use:
    1. Run your training experiment to get best config
    2. Set BEST_CONFIG below (or it auto-picks from dense_results.csv)
    3. Run:  python fault_predictor_app.py
    4. Open: http://localhost:5000
"""

import os
import io
import json
import warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, accuracy_score, precision_score,
    recall_score, f1_score, log_loss, confusion_matrix
)
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier          # ← Random Forest
import joblib

from flask import Flask, request, jsonify, render_template_string

# ─────────────────────────────────────────────
# CONFIG — edit to match your best experiment result
# ─────────────────────────────────────────────
DATA_PATH    = "synthetic_nim_parallel_10000.csv"
RESULTS_CSV  = "results/synthetic/dense_results.csv"   # from your experiment
MODEL_PATH   = "faultsense_model.joblib"

DROP_COLS    = ["location"]
TARGET       = "faulty"
CAT_COLS     = ["equipment"]
NUM_COLS     = ["temperature", "pressure", "vibration", "humidity"]
RANDOM_STATE = 42
THRESHOLD    = 0.5

# Fixed RF params (analogous to FIXED_PARAMS in LightGBM version)
FIXED_PARAMS = dict(
    max_depth        = 8,
    min_samples_leaf = 20,       # ≈ min_child_samples
    max_samples      = 0.8,      # ≈ subsample (bootstrap fraction)
    class_weight     = "balanced",
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
)

# If you already know your best config, set it here.
# Otherwise set BEST_CONFIG = None and it reads from RESULTS_CSV.
BEST_CONFIG = None
# Example:
BEST_CONFIG = {
    "max_features": "sqrt",
    "n_estimators": 200,
    "train_ratio": 0.70,
    "val_ratio": 0.20,
    "test_ratio": 0.10,
}

# ─────────────────────────────────────────────
# MODEL TRAINING / LOADING
# ─────────────────────────────────────────────
def make_preprocessor():
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_COLS),
        ("num", "passthrough", NUM_COLS),
    ])


def load_best_config():
    """Read best config from experiment results CSV."""
    if not os.path.exists(RESULTS_CSV):
        raise FileNotFoundError(
            f"Results file not found: {RESULTS_CSV}\n"
            "Run your training experiment first, or set BEST_CONFIG manually."
        )
    df = pd.read_csv(RESULTS_CSV)
    df = df.sort_values("val_f1", ascending=False)
    row = df.iloc[0]
    return {
        "max_features" : str(row["max_features"]),   # ← replaces learning_rate
        "n_estimators" : int(row["n_estimators"]),
        "train_ratio"  : float(row["train_ratio"]),
        "val_ratio"    : float(row["val_ratio"]),
        "test_ratio"   : float(row["test_ratio"]),
        "val_f1"       : float(row["val_f1"]),
        "val_auc"      : float(row["val_auc"]),
        "val_accuracy" : float(row["val_accuracy"]),
        "val_precision": float(row["val_precision"]),
        "val_recall"   : float(row["val_recall"]),
        "val_logloss"  : float(row["val_logloss"]),
        "split_label"  : str(row["split_label"]),
    }


def train_model(cfg: dict):
    """Train the model with the given best config and save it."""
    print(f"\n🔧  Training with config: max_features={cfg['max_features']}, "
          f"n_est={cfg['n_estimators']}, split={cfg['train_ratio']}/{cfg['val_ratio']}/{cfg['test_ratio']}")

    df_raw = pd.read_csv(DATA_PATH)
    df_raw = df_raw.drop(columns=DROP_COLS, errors="ignore")
    X = df_raw.drop(columns=[TARGET])
    y = df_raw[TARGET]

    train_r, val_r, test_r = cfg["train_ratio"], cfg["val_ratio"], cfg["test_ratio"]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_r, stratify=y, random_state=RANDOM_STATE
    )
    val_relative = val_r / (train_r + val_r)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_relative,
        stratify=y_trainval, random_state=RANDOM_STATE
    )

    # Parse max_features: convert numeric strings back to float
    mf = cfg["max_features"]
    try:
        mf = float(mf)
    except ValueError:
        pass   # keep as string e.g. "sqrt", "log2"

    pre = make_preprocessor()
    clf = RandomForestClassifier(               # ← Random Forest
        n_estimators = cfg["n_estimators"],
        max_features = mf,
        **FIXED_PARAMS
    )
    pipeline = Pipeline([("pre", pre), ("clf", clf)])
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    y_prob_test = pipeline.predict_proba(X_test)[:, 1]
    y_pred_test = (y_prob_test >= THRESHOLD).astype(int)
    test_metrics = {
        "test_auc"      : round(roc_auc_score(y_test, y_prob_test), 4),
        "test_accuracy" : round(accuracy_score(y_test, y_pred_test), 4),
        "test_precision": round(precision_score(y_test, y_pred_test, zero_division=0), 4),
        "test_recall"   : round(recall_score(y_test, y_pred_test, zero_division=0), 4),
        "test_f1"       : round(f1_score(y_test, y_pred_test, zero_division=0), 4),
        "test_logloss"  : round(log_loss(y_test, y_prob_test), 4),
    }
    cm = confusion_matrix(y_test, y_pred_test).tolist()

    # Save
    artifact = {"pipeline": pipeline, "config": cfg, "test_metrics": test_metrics, "cm": cm}
    joblib.dump(artifact, MODEL_PATH)
    print(f"✅  Model saved → {MODEL_PATH}")
    print(f"    Test AUC={test_metrics['test_auc']}  F1={test_metrics['test_f1']}")
    return artifact


def load_or_train():
    if os.path.exists(MODEL_PATH):
        print(f"✅  Loading saved model from {MODEL_PATH}")
        return joblib.load(MODEL_PATH)

    cfg = BEST_CONFIG if BEST_CONFIG else load_best_config()
    return train_model(cfg)


# ─────────────────────────────────────────────
# FLASK APP
# ─────────────────────────────────────────────
app = Flask(__name__)
ARTIFACT = None   # loaded at startup

EQUIPMENT_OPTIONS = ["pump", "compressor", "motor", "valve", "sensor"]  # adjust to your data

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>FaultSense — Equipment Fault Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:ital,wght@0,300;0,500;0,700;1,300&display=swap" rel="stylesheet">
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

  :root {
    --bg: #0a0c10;
    --surface: #111318;
    --surface2: #181c24;
    --border: #232838;
    --accent: #00e5a0;
    --accent2: #ff4d6d;
    --accent3: #4d9fff;
    --text: #e8eaf0;
    --muted: #6b7280;
    --mono: 'Space Mono', monospace;
    --sans: 'DM Sans', sans-serif;
  }

  html { font-size: 16px; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: var(--sans);
    min-height: 100vh;
    overflow-x: hidden;
  }

  body::before {
    content: '';
    position: fixed; inset: 0;
    background-image:
      linear-gradient(rgba(0,229,160,.04) 1px, transparent 1px),
      linear-gradient(90deg, rgba(0,229,160,.04) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
  }

  .blob {
    position: fixed;
    width: 600px; height: 600px;
    border-radius: 50%;
    filter: blur(120px);
    opacity: .15;
    pointer-events: none;
    animation: drift 12s ease-in-out infinite alternate;
    z-index: 0;
  }
  .blob-1 { background: var(--accent);  top: -200px; left: -200px; }
  .blob-2 { background: var(--accent3); bottom: -200px; right: -100px; animation-delay: -6s; }
  @keyframes drift { from { transform: translate(0,0) scale(1); } to { transform: translate(40px,30px) scale(1.05); } }

  .wrapper {
    position: relative; z-index: 1;
    max-width: 1100px;
    margin: 0 auto;
    padding: 40px 24px 80px;
  }

  header {
    display: flex; align-items: center; gap: 16px;
    margin-bottom: 48px;
    border-bottom: 1px solid var(--border);
    padding-bottom: 24px;
  }
  .logo-mark {
    width: 44px; height: 44px;
    background: var(--accent);
    border-radius: 10px;
    display: grid; place-items: center;
    font-family: var(--mono);
    font-weight: 700;
    font-size: 18px;
    color: var(--bg);
    flex-shrink: 0;
    box-shadow: 0 0 24px rgba(0,229,160,.4);
  }
  header h1 {
    font-family: var(--mono);
    font-size: 1.5rem;
    letter-spacing: -.5px;
    color: var(--text);
  }
  header p { font-size: .85rem; color: var(--muted); margin-top: 2px; }
  .badge {
    margin-left: auto;
    font-family: var(--mono);
    font-size: .7rem;
    background: rgba(0,229,160,.12);
    color: var(--accent);
    border: 1px solid rgba(0,229,160,.3);
    border-radius: 6px;
    padding: 4px 10px;
    white-space: nowrap;
  }

  .main-grid {
    display: grid;
    grid-template-columns: 1fr 380px;
    gap: 24px;
    align-items: start;
  }
  @media (max-width: 860px) { .main-grid { grid-template-columns: 1fr; } }

  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 28px;
  }
  .card-title {
    font-family: var(--mono);
    font-size: .75rem;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    color: var(--muted);
    margin-bottom: 20px;
    display: flex; align-items: center; gap: 8px;
  }
  .card-title::before {
    content: '';
    display: inline-block;
    width: 6px; height: 6px;
    background: var(--accent);
    border-radius: 50%;
    box-shadow: 0 0 8px var(--accent);
  }

  .form-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 16px;
  }
  @media (max-width: 560px) { .form-grid { grid-template-columns: 1fr; } }

  .field { display: flex; flex-direction: column; gap: 8px; }
  .field label {
    font-size: .78rem;
    font-family: var(--mono);
    color: var(--muted);
    letter-spacing: .5px;
  }
  .field input, .field select {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    color: var(--text);
    font-family: var(--mono);
    font-size: .9rem;
    padding: 11px 14px;
    outline: none;
    transition: border-color .2s, box-shadow .2s;
    -webkit-appearance: none;
    appearance: none;
  }
  .field input:focus, .field select:focus {
    border-color: var(--accent);
    box-shadow: 0 0 0 3px rgba(0,229,160,.15);
  }
  .field select {
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b7280' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat;
    background-position: right 14px center;
    padding-right: 36px;
    cursor: pointer;
  }

  .slider-wrap { display: flex; flex-direction: column; gap: 6px; }
  .slider-row { display: flex; align-items: center; gap: 10px; }
  input[type=range] {
    flex: 1;
    -webkit-appearance: none;
    height: 4px;
    border-radius: 4px;
    background: var(--border);
    outline: none;
    cursor: pointer;
  }
  input[type=range]::-webkit-slider-thumb {
    -webkit-appearance: none;
    width: 16px; height: 16px;
    border-radius: 50%;
    background: var(--accent);
    box-shadow: 0 0 8px rgba(0,229,160,.5);
    transition: transform .15s;
  }
  input[type=range]::-webkit-slider-thumb:hover { transform: scale(1.3); }
  .slider-val {
    font-family: var(--mono);
    font-size: .85rem;
    color: var(--accent);
    min-width: 52px;
    text-align: right;
  }

  .btn-predict {
    margin-top: 24px;
    width: 100%;
    padding: 14px;
    background: var(--accent);
    color: var(--bg);
    border: none;
    border-radius: 12px;
    font-family: var(--mono);
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 1px;
    cursor: pointer;
    transition: transform .15s, box-shadow .2s, background .2s;
    box-shadow: 0 0 20px rgba(0,229,160,.3);
  }
  .btn-predict:hover { transform: translateY(-2px); box-shadow: 0 0 32px rgba(0,229,160,.5); }
  .btn-predict:active { transform: translateY(0); }
  .btn-predict:disabled { background: var(--muted); cursor: not-allowed; box-shadow: none; transform: none; }

  .result-card {
    border-radius: 16px;
    padding: 28px;
    border: 1px solid var(--border);
    background: var(--surface);
    transition: border-color .4s;
  }
  .result-card.faulty  { border-color: var(--accent2); background: rgba(255,77,109,.06); }
  .result-card.healthy { border-color: var(--accent);  background: rgba(0,229,160,.06); }

  .verdict {
    font-family: var(--mono);
    font-size: 2rem;
    font-weight: 700;
    letter-spacing: -1px;
    margin-bottom: 6px;
  }
  .verdict.faulty  { color: var(--accent2); text-shadow: 0 0 20px rgba(255,77,109,.4); }
  .verdict.healthy { color: var(--accent);  text-shadow: 0 0 20px rgba(0,229,160,.4); }
  .verdict-sub { font-size: .85rem; color: var(--muted); margin-bottom: 24px; }

  .prob-bar-wrap { margin-bottom: 24px; }
  .prob-label { font-family: var(--mono); font-size: .72rem; color: var(--muted); margin-bottom: 6px; display: flex; justify-content: space-between; }
  .prob-track {
    height: 10px;
    background: var(--border);
    border-radius: 10px;
    overflow: hidden;
  }
  .prob-fill {
    height: 100%;
    border-radius: 10px;
    transition: width .6s cubic-bezier(.4,0,.2,1);
  }
  .prob-fill.faulty  { background: linear-gradient(90deg, #ff4d6d, #ff8fa3); }
  .prob-fill.healthy { background: linear-gradient(90deg, #00e5a0, #5eead4); }

  .mini-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
  .mini-metric {
    background: var(--surface2);
    border-radius: 10px;
    padding: 12px;
    border: 1px solid var(--border);
  }
  .mini-metric .mm-val {
    font-family: var(--mono);
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--accent3);
  }
  .mini-metric .mm-key {
    font-size: .7rem;
    color: var(--muted);
    margin-top: 2px;
    font-family: var(--mono);
    letter-spacing: .5px;
  }

  .info-panel { margin-top: 24px; }
  .info-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 9px 0;
    border-bottom: 1px solid var(--border);
    font-size: .82rem;
  }
  .info-row:last-child { border-bottom: none; }
  .info-key { color: var(--muted); font-family: var(--mono); font-size: .72rem; }
  .info-val { font-family: var(--mono); color: var(--text); font-weight: 700; }
  .info-val.green { color: var(--accent); }

  .history-list { max-height: 260px; overflow-y: auto; display: flex; flex-direction: column; gap: 8px; }
  .history-list::-webkit-scrollbar { width: 4px; }
  .history-list::-webkit-scrollbar-track { background: var(--surface2); }
  .history-list::-webkit-scrollbar-thumb { background: var(--border); border-radius: 2px; }

  .hist-item {
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 14px;
    display: flex; justify-content: space-between; align-items: center;
    font-size: .78rem;
    animation: slideIn .3s ease;
  }
  @keyframes slideIn { from { opacity: 0; transform: translateY(-6px); } to { opacity: 1; transform: translateY(0); } }
  .hist-item .hist-equip { color: var(--muted); font-family: var(--mono); font-size: .7rem; }
  .hist-badge {
    font-family: var(--mono);
    font-size: .68rem;
    padding: 3px 8px;
    border-radius: 6px;
    font-weight: 700;
  }
  .hist-badge.faulty  { background: rgba(255,77,109,.2); color: var(--accent2); }
  .hist-badge.healthy { background: rgba(0,229,160,.2); color: var(--accent); }

  .spinner {
    display: none;
    width: 20px; height: 20px;
    border: 2px solid rgba(10,12,16,.3);
    border-top-color: var(--bg);
    border-radius: 50%;
    animation: spin .6s linear infinite;
    margin: 0 auto;
  }
  @keyframes spin { to { transform: rotate(360deg); } }
  .btn-predict.loading .btn-text { display: none; }
  .btn-predict.loading .spinner { display: block; }

  .idle-state {
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    gap: 12px; padding: 32px 0; color: var(--muted); text-align: center;
  }
  .idle-icon { font-size: 2.5rem; opacity: .4; }
  .idle-state p { font-size: .85rem; line-height: 1.6; }

  .toast {
    position: fixed; bottom: 24px; right: 24px;
    background: var(--surface2); border: 1px solid var(--accent2);
    color: var(--accent2); border-radius: 10px; padding: 12px 18px;
    font-family: var(--mono); font-size: .8rem;
    transform: translateY(80px); opacity: 0;
    transition: all .3s; z-index: 999;
  }
  .toast.show { transform: translateY(0); opacity: 1; }
</style>
</head>
<body>
<div class="blob blob-1"></div>
<div class="blob blob-2"></div>

<div class="wrapper">
  <header>
    <div class="logo-mark">FS</div>
    <div>
      <h1>FaultSense</h1>
      <p>Random Forest Equipment Fault Predictor</p>
    </div>
    <div class="badge" id="model-badge">Model Loading…</div>
  </header>

  <div class="main-grid">
    <div style="display:flex;flex-direction:column;gap:20px;">

      <div class="card">
        <div class="card-title">Sensor Readings</div>
        <div class="form-grid">

          <div class="field">
            <label>Equipment Type</label>
            <select id="equipment">
              <option value="">— Select Equipment —</option>
            </select>
          </div>

          <div class="field slider-wrap">
            <label>Temperature (°C)</label>
            <div class="slider-row">
              <input type="range" id="temperature" min="-20" max="120" step="0.5" value="40"
                     oninput="document.getElementById('temperature-val').textContent=parseFloat(this.value).toFixed(1)+'°C'">
              <span class="slider-val" id="temperature-val">40.0°C</span>
            </div>
          </div>

          <div class="field slider-wrap">
            <label>Pressure (bar)</label>
            <div class="slider-row">
              <input type="range" id="pressure" min="0" max="20" step="0.1" value="5"
                     oninput="document.getElementById('pressure-val').textContent=parseFloat(this.value).toFixed(1)+' bar'">
              <span class="slider-val" id="pressure-val">5.0 bar</span>
            </div>
          </div>

          <div class="field slider-wrap">
            <label>Vibration (mm/s)</label>
            <div class="slider-row">
              <input type="range" id="vibration" min="0" max="50" step="0.1" value="5"
                     oninput="document.getElementById('vibration-val').textContent=parseFloat(this.value).toFixed(1)+' mm/s'">
              <span class="slider-val" id="vibration-val">5.0 mm/s</span>
            </div>
          </div>

          <div class="field slider-wrap" style="grid-column:1/-1">
            <label>Humidity (%)</label>
            <div class="slider-row">
              <input type="range" id="humidity" min="0" max="100" step="1" value="50"
                     oninput="document.getElementById('humidity-val').textContent=parseInt(this.value)+'%'">
              <span class="slider-val" id="humidity-val">50%</span>
            </div>
          </div>

        </div>
        <button class="btn-predict" id="predict-btn" onclick="predict()">
          <span class="btn-text">⚡ Run Prediction</span>
          <div class="spinner"></div>
        </button>
      </div>

      <div class="card">
        <div class="card-title">Prediction History</div>
        <div class="history-list" id="history-list">
          <div style="color:var(--muted);font-size:.8rem;font-family:var(--mono);text-align:center;padding:16px 0;">
            No predictions yet
          </div>
        </div>
      </div>
    </div>

    <div style="display:flex;flex-direction:column;gap:20px;">

      <div class="result-card" id="result-card">
        <div class="idle-state" id="idle-state">
          <div class="idle-icon">🔬</div>
          <p>Enter sensor readings<br>and run a prediction<br>to see results here.</p>
        </div>
        <div id="result-content" style="display:none;">
          <div class="verdict" id="verdict-text"></div>
          <div class="verdict-sub" id="verdict-sub"></div>
          <div class="prob-bar-wrap">
            <div class="prob-label">
              <span>Fault Probability</span>
              <span id="prob-pct"></span>
            </div>
            <div class="prob-track">
              <div class="prob-fill" id="prob-fill" style="width:0%"></div>
            </div>
          </div>
          <div class="mini-metrics" id="mini-metrics"></div>
        </div>
      </div>

      <div class="card">
        <div class="card-title">Model Configuration</div>
        <div class="info-panel" id="model-info">
          <div style="color:var(--muted);font-size:.8rem;font-family:var(--mono);text-align:center;padding:12px 0;">
            Loading…
          </div>
        </div>
      </div>

    </div>
  </div>
</div>

<div class="toast" id="toast"></div>

<script>
  let history = [];

  async function loadModelInfo() {
    try {
      const res = await fetch('/model_info');
      const data = await res.json();
      if (data.error) { showToast('Model not ready: ' + data.error); return; }

      if (data.equipment_options && data.equipment_options.length) {
        const sel = document.getElementById('equipment');
        sel.innerHTML = data.equipment_options.map(
          e => `<option value="${e}">${e.charAt(0).toUpperCase() + e.slice(1)}</option>`
        ).join('');
      }

      // Badge now shows max_features instead of learning_rate
      document.getElementById('model-badge').textContent =
        `max_feat=${data.config.max_features} · N=${data.config.n_estimators}`;

      const rows = [
        ['Max Features',   data.config.max_features],   // ← replaces Learning Rate
        ['N Estimators',   data.config.n_estimators],
        ['Split',          data.config.split_label || `${data.config.train_ratio}/${data.config.val_ratio}/${data.config.test_ratio}`],
        ['Test AUC',       (data.test_metrics.test_auc*100).toFixed(2)+'%'],
        ['Test F1',        (data.test_metrics.test_f1*100).toFixed(2)+'%'],
        ['Test Accuracy',  (data.test_metrics.test_accuracy*100).toFixed(2)+'%'],
        ['Test Precision', (data.test_metrics.test_precision*100).toFixed(2)+'%'],
        ['Test Recall',    (data.test_metrics.test_recall*100).toFixed(2)+'%'],
      ];

      document.getElementById('model-info').innerHTML = rows.map(([k,v],i) => `
        <div class="info-row">
          <span class="info-key">${k}</span>
          <span class="info-val ${i>=3?'green':''}">${v}</span>
        </div>`).join('');
    } catch(e) {
      document.getElementById('model-badge').textContent = 'Model Error';
    }
  }

  async function predict() {
    const btn = document.getElementById('predict-btn');
    btn.classList.add('loading');
    btn.disabled = true;

    const payload = {
      equipment:   document.getElementById('equipment').value,
      temperature: parseFloat(document.getElementById('temperature').value),
      pressure:    parseFloat(document.getElementById('pressure').value),
      vibration:   parseFloat(document.getElementById('vibration').value),
      humidity:    parseFloat(document.getElementById('humidity').value),
    };

    try {
      const res = await fetch('/predict', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(payload)
      });
      const data = await res.json();
      if (data.error) { showToast('Error: '+data.error); return; }
      showResult(data, payload);
      addHistory(data, payload);
    } catch(e) {
      showToast('Network error — is the server running?');
    } finally {
      btn.classList.remove('loading');
      btn.disabled = false;
    }
  }

  function showResult(data, payload) {
    const isFaulty = data.prediction === 1;
    const prob = (data.probability * 100).toFixed(1);
    const cls = isFaulty ? 'faulty' : 'healthy';

    const card = document.getElementById('result-card');
    card.className = 'result-card ' + cls;

    document.getElementById('idle-state').style.display = 'none';
    const rc = document.getElementById('result-content');
    rc.style.display = 'block';

    const vt = document.getElementById('verdict-text');
    vt.className = 'verdict ' + cls;
    vt.textContent = isFaulty ? '⚠ FAULT DETECTED' : '✓ HEALTHY';

    document.getElementById('verdict-sub').textContent =
      isFaulty
        ? `High fault probability — immediate inspection recommended.`
        : `Equipment readings within normal operating range.`;

    document.getElementById('prob-pct').textContent = prob + '%';
    const fill = document.getElementById('prob-fill');
    fill.className = 'prob-fill ' + cls;
    setTimeout(() => fill.style.width = prob + '%', 50);

    document.getElementById('mini-metrics').innerHTML = [
      ['Probability', prob+'%'],
      ['Confidence',  data.confidence],
      ['Equipment',   payload.equipment],
      ['Threshold',   (data.threshold*100).toFixed(0)+'%'],
    ].map(([k,v]) => `
      <div class="mini-metric">
        <div class="mm-val">${v}</div>
        <div class="mm-key">${k}</div>
      </div>`).join('');
  }

  function addHistory(data, payload) {
    const isFaulty = data.prediction === 1;
    const cls = isFaulty ? 'faulty' : 'healthy';
    const item = document.createElement('div');
    item.className = 'hist-item';
    item.innerHTML = `
      <div>
        <div style="font-family:var(--mono);font-size:.78rem;">${payload.equipment}</div>
        <div class="hist-equip">T=${payload.temperature}° P=${payload.pressure}bar V=${payload.vibration}</div>
      </div>
      <span class="hist-badge ${cls}">${isFaulty ? 'FAULT' : 'OK'} · ${(data.probability*100).toFixed(1)}%</span>`;

    const list = document.getElementById('history-list');
    if (list.children.length === 1 && list.children[0].style.color === 'var(--muted)') {
      list.innerHTML = '';
    }
    list.prepend(item);
    if (list.children.length > 20) list.removeChild(list.lastChild);
  }

  function showToast(msg) {
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 3500);
  }

  loadModelInfo();
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/model_info")
def model_info():
    if ARTIFACT is None:
        return jsonify({"error": "Model not loaded"})
    cfg = ARTIFACT["config"]
    return jsonify({
        "config"            : cfg,
        "test_metrics"      : ARTIFACT["test_metrics"],
        "cm"                : ARTIFACT["cm"],
        "equipment_options" : EQUIPMENT_OPTIONS,
    })


@app.route("/predict", methods=["POST"])
def predict():
    if ARTIFACT is None:
        return jsonify({"error": "Model not loaded"}), 503

    body = request.get_json(force=True)
    try:
        row = pd.DataFrame([{
            "equipment"  : body["equipment"],
            "temperature": float(body["temperature"]),
            "pressure"   : float(body["pressure"]),
            "vibration"  : float(body["vibration"]),
            "humidity"   : float(body["humidity"]),
        }])
    except (KeyError, ValueError) as e:
        return jsonify({"error": f"Bad input: {e}"}), 400

    pipeline = ARTIFACT["pipeline"]
    prob = float(pipeline.predict_proba(row)[0, 1])
    pred = int(prob >= THRESHOLD)

    if prob > 0.85 or prob < 0.15:
        confidence = "HIGH"
    elif prob > 0.65 or prob < 0.35:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return jsonify({
        "prediction" : pred,
        "probability": round(prob, 4),
        "confidence" : confidence,
        "threshold"  : THRESHOLD,
        "label"      : "FAULTY" if pred == 1 else "HEALTHY",
    })


# ─────────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 56)
    print("  FaultSense — Random Forest Equipment Fault Predictor")
    print("=" * 56)

    ARTIFACT = load_or_train()

    cfg = ARTIFACT["config"]
    tm  = ARTIFACT["test_metrics"]
    print(f"\n  Config  : max_features={cfg['max_features']}, n_est={cfg['n_estimators']}")
    print(f"  Test AUC: {tm['test_auc']}  |  F1: {tm['test_f1']}  |  Acc: {tm['test_accuracy']}")
    print(f"\n🌐 Open http://localhost:5000 in your browser\n")

    app.run(debug=False, host="0.0.0.0", port=5000)
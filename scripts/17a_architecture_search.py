"""
S175 Surrogate Model — Step 5: Model Complexity Tuning
========================================================
Tune the MLP architecture for:
  1. Regressor (10 outputs) — primary focus
  2. Classifier 1 (complete infeasibility)
  3. Classifier 2 (fuel infeasibility)

Test different:
  - Hidden layer architectures
  - Learning rates
  - Batch sizes
"""

import pandas as pd
import numpy as np
import time
import os
import json
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, confusion_matrix
)

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = '/home/macierz/mohabdal/S175_sample_5pct_stratified.csv'
OUTPUT_DIR = '/home/macierz/mohabdal/S175_experiments'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FEATURES = ['draft', 'trim', 'rpm', 'PD', 'Hs', 'Tp', 'Chi', 'Vwind', 'Theta_wind']
OUTPUT_COLS = ['speed', 'power', 'torque', 'lat_acc', 'MSI',
              'roll', 'slam', 'green_water', 'prop_emerg', 'fuel']
OUTPUT_UNITS = {
    'speed': 'kn', 'power': 'kW', 'torque': 'kNm',
    'lat_acc': 'g', 'MSI': '%', 'roll': 'deg',
    'slam': '%', 'green_water': '%', 'prop_emerg': '%', 'fuel': 'kg/nm'
}

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# HYPERPARAMETER CONFIGURATIONS TO TEST
# ============================================================

# Regressor architectures (from simple to complex)
REG_CONFIGS = {
    'Small (128-64)': {
        'hidden_layer_sizes': (128, 64),
        'learning_rate_init': 0.001,
        'batch_size': 1024
    },
    'Medium (256-128-64)': {
        'hidden_layer_sizes': (256, 128, 64),
        'learning_rate_init': 0.001,
        'batch_size': 1024
    },
    'Large (512-256-128)': {
        'hidden_layer_sizes': (512, 256, 128),
        'learning_rate_init': 0.001,
        'batch_size': 1024
    },
    'XLarge (512-256-128-64)': {
        'hidden_layer_sizes': (512, 256, 128, 64),
        'learning_rate_init': 0.001,
        'batch_size': 1024
    },
    'Wide (512-512)': {
        'hidden_layer_sizes': (512, 512),
        'learning_rate_init': 0.001,
        'batch_size': 1024
    },
    'Medium-LR0.0005': {
        'hidden_layer_sizes': (256, 128, 64),
        'learning_rate_init': 0.0005,
        'batch_size': 1024
    },
    'Large-LR0.0005': {
        'hidden_layer_sizes': (512, 256, 128),
        'learning_rate_init': 0.0005,
        'batch_size': 1024
    },
    'Large-Batch2048': {
        'hidden_layer_sizes': (512, 256, 128),
        'learning_rate_init': 0.001,
        'batch_size': 2048
    },
    'Large-Batch512': {
        'hidden_layer_sizes': (512, 256, 128),
        'learning_rate_init': 0.001,
        'batch_size': 512
    },
}

# Classifier architectures (fewer configs since already good)
CLF_CONFIGS = {
    'Medium (256-128-64)': {
        'hidden_layer_sizes': (256, 128, 64),
        'learning_rate_init': 0.001,
        'batch_size': 1024
    },
    'Large (512-256-128)': {
        'hidden_layer_sizes': (512, 256, 128),
        'learning_rate_init': 0.001,
        'batch_size': 1024
    },
    'XLarge (512-256-128-64)': {
        'hidden_layer_sizes': (512, 256, 128, 64),
        'learning_rate_init': 0.001,
        'batch_size': 1024
    },
}

# ============================================================
# STEP 1: LOAD AND PREPARE DATA
# ============================================================
print("=" * 70)
print("S175 SURROGATE MODEL — STEP 5: MODEL COMPLEXITY TUNING")
print("=" * 70)

print(f"\n[1/5] Loading and preparing data")
t0 = time.time()
df = pd.read_csv(DATA_PATH, sep=';')
print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

# Row categories
all_neg = (df[OUTPUT_COLS] == -1).all(axis=1)
fuel_neg_only = (df['fuel'] == -1) & ~all_neg
fully_valid = ~all_neg & ~fuel_neg_only

X = df[INPUT_FEATURES].values

# Labels
label_all_infeasible = all_neg.astype(int).values
label_fuel_infeasible = fuel_neg_only.astype(int).values

# Stratification
hs_bins = np.digitize(df['Hs'].values, bins=[2.01, 5.01, 8.01])
vwind_bins = np.digitize(df['Vwind'].values, bins=[7.5, 17.5])
status = np.zeros(len(df), dtype=int)
status[fuel_neg_only.values] = 1
status[all_neg.values] = 2
strata = hs_bins * 100 + vwind_bins * 10 + status

# Split
indices = np.arange(len(df))
idx_train, idx_temp, _, strata_temp = train_test_split(
    indices, strata, test_size=0.2, random_state=RANDOM_STATE, stratify=strata
)
idx_val, idx_test = train_test_split(
    idx_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=strata_temp
)

X_train, X_val, X_test = X[idx_train], X[idx_val], X[idx_test]

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Prepare datasets for each component
# Classifier 1: all data
y_clf1_train = label_all_infeasible[idx_train]
y_clf1_val = label_all_infeasible[idx_val]
y_clf1_test = label_all_infeasible[idx_test]

# Classifier 2: non-completely-infeasible rows
not_all_neg_train = ~all_neg.values[idx_train]
not_all_neg_test = ~all_neg.values[idx_test]

X_train_clf2 = X_train_scaled[not_all_neg_train]
X_test_clf2 = X_test_scaled[not_all_neg_test]
y_clf2_train = fuel_neg_only.values[idx_train][not_all_neg_train].astype(int)
y_clf2_test = fuel_neg_only.values[idx_test][not_all_neg_test].astype(int)

# Regressor: fully valid rows only
fv_train = fully_valid.values[idx_train]
fv_val = fully_valid.values[idx_val]
fv_test = fully_valid.values[idx_test]

X_train_reg = X_train_scaled[fv_train]
X_val_reg = X_val_scaled[fv_val]
X_test_reg = X_test_scaled[fv_test]

y_train_reg = df[OUTPUT_COLS].values[idx_train][fv_train]
y_val_reg = df[OUTPUT_COLS].values[idx_val][fv_val]
y_test_reg = df[OUTPUT_COLS].values[idx_test][fv_test]

# Scale outputs
output_scaler = StandardScaler()
y_train_reg_scaled = output_scaler.fit_transform(y_train_reg)

print(f"  Classifier 1 training: {len(X_train):,} rows")
print(f"  Classifier 2 training: {len(X_train_clf2):,} rows")
print(f"  Regressor training: {len(X_train_reg):,} rows")
print(f"  Test set: {len(idx_test):,} rows")

# ============================================================
# STEP 2: TUNE REGRESSOR (10 outputs)
# ============================================================
print(f"\n[2/5] TUNING REGRESSOR (10 outputs)")
print(f"  Testing {len(REG_CONFIGS)} configurations")

reg_results = {}

for config_name, config in REG_CONFIGS.items():
    print(f"\n  --- {config_name} ---")
    print(f"  Architecture: {config['hidden_layer_sizes']}, "
          f"LR: {config['learning_rate_init']}, Batch: {config['batch_size']}")
    
    t0 = time.time()
    model = MLPRegressor(
        hidden_layer_sizes=config['hidden_layer_sizes'],
        activation='relu',
        solver='adam',
        learning_rate='adaptive',
        learning_rate_init=config['learning_rate_init'],
        max_iter=300,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=15,
        batch_size=config['batch_size'],
        random_state=RANDOM_STATE,
        verbose=False
    )
    model.fit(X_train_reg, y_train_reg_scaled)
    train_time = time.time() - t0
    
    # Predict on validation and test
    pred_val_scaled = model.predict(X_val_reg)
    pred_test_scaled = model.predict(X_test_reg)
    pred_val = output_scaler.inverse_transform(pred_val_scaled)
    pred_test = output_scaler.inverse_transform(pred_test_scaled)
    
    print(f"  Training time: {train_time:.1f}s (iterations: {model.n_iter_})")
    print(f"  Final loss: {model.loss_:.6f}")
    
    # Per-output metrics on test
    output_metrics = {}
    total_mae = 0
    total_r2 = 0
    total_overest = 0
    
    for i, col in enumerate(OUTPUT_COLS):
        y_true = y_test_reg[:, i]
        y_pred = pred_test[:, i]
        
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        max_err = np.max(np.abs(y_true - y_pred))
        overest = 100 * (y_pred > y_true).sum() / len(y_true)
        
        output_metrics[col] = {
            'MAE': float(mae), 'R2': float(r2),
            'Max_Error': float(max_err), 'Overest_pct': float(overest)
        }
        total_mae += mae
        total_r2 += r2
        total_overest += overest
    
    # Validation metrics for overfitting check
    val_output_metrics = {}
    for i, col in enumerate(OUTPUT_COLS):
        y_true_v = y_val_reg[:, i]
        y_pred_v = pred_val[:, i]
        val_output_metrics[col] = {
            'MAE': float(mean_absolute_error(y_true_v, y_pred_v)),
            'R2': float(r2_score(y_true_v, y_pred_v))
        }
    
    avg_mae = total_mae / 10
    avg_r2 = total_r2 / 10
    avg_overest = total_overest / 10
    
    # Overfitting check: compare val vs test
    val_avg_mae = np.mean([val_output_metrics[c]['MAE'] for c in OUTPUT_COLS])
    
    reg_results[config_name] = {
        'config': {k: str(v) for k, v in config.items()},
        'train_time_s': float(train_time),
        'iterations': int(model.n_iter_),
        'final_loss': float(model.loss_),
        'avg_MAE': float(avg_mae),
        'avg_R2': float(avg_r2),
        'avg_Overest_pct': float(avg_overest),
        'val_avg_MAE': float(val_avg_mae),
        'per_output': output_metrics
    }
    
    # Key outputs summary
    print(f"  Speed:  MAE={output_metrics['speed']['MAE']:.4f}, R²={output_metrics['speed']['R2']:.6f}")
    print(f"  Fuel:   MAE={output_metrics['fuel']['MAE']:.4f}, R²={output_metrics['fuel']['R2']:.6f}")
    print(f"  Avg:    MAE={avg_mae:.4f}, R²={avg_r2:.6f}, Overest={avg_overest:.1f}%")
    print(f"  Val MAE={val_avg_mae:.4f} vs Test MAE={avg_mae:.4f} "
          f"(diff={abs(val_avg_mae-avg_mae):.4f}) {'✓ No overfit' if abs(val_avg_mae-avg_mae)/avg_mae < 0.1 else '⚠ Possible overfit'}")

# Regressor ranking
print(f"\n{'='*70}")
print(f"REGRESSOR RANKING (by average MAE)")
print(f"{'='*70}")
print(f"\n{'Rank':<6} {'Config':<30} {'Avg MAE':<12} {'Speed MAE':<12} {'Fuel MAE':<12} "
      f"{'Avg R²':<12} {'Time(s)':<10} {'Iters':<8}")
print(f"{'-'*102}")

sorted_reg = sorted(reg_results.items(), key=lambda x: x[1]['avg_MAE'])
for rank, (name, r) in enumerate(sorted_reg, 1):
    print(f"{rank:<6} {name:<30} {r['avg_MAE']:<12.4f} "
          f"{r['per_output']['speed']['MAE']:<12.4f} "
          f"{r['per_output']['fuel']['MAE']:<12.4f} "
          f"{r['avg_R2']:<12.6f} {r['train_time_s']:<10.1f} {r['iterations']:<8}")

best_reg_config = sorted_reg[0][0]
print(f"\n→ Best regressor config: {best_reg_config}")

# ============================================================
# STEP 3: TUNE CLASSIFIERS
# ============================================================
print(f"\n[3/5] TUNING CLASSIFIERS")

# --- Classifier 1 ---
print(f"\n  === CLASSIFIER 1 (complete infeasibility) ===")
clf1_results = {}

for config_name, config in CLF_CONFIGS.items():
    print(f"\n  --- {config_name} ---")
    
    t0 = time.time()
    model = MLPClassifier(
        hidden_layer_sizes=config['hidden_layer_sizes'],
        activation='relu', solver='adam',
        learning_rate='adaptive',
        learning_rate_init=config['learning_rate_init'],
        max_iter=300, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15,
        batch_size=config['batch_size'],
        random_state=RANDOM_STATE, verbose=False
    )
    model.fit(X_train_scaled, y_clf1_train)
    train_time = time.time() - t0
    
    pred_test = model.predict(X_test_scaled)
    cm = confusion_matrix(y_clf1_test, pred_test)
    f1 = f1_score(y_clf1_test, pred_test)
    acc = accuracy_score(y_clf1_test, pred_test)
    
    false_neg = cm[1, 0]  # True infeasible, predicted feasible (DANGEROUS)
    
    clf1_results[config_name] = {
        'accuracy': float(acc), 'f1': float(f1),
        'false_negatives': int(false_neg),
        'train_time_s': float(train_time),
        'iterations': int(model.n_iter_)
    }
    
    print(f"  Time: {train_time:.1f}s, Iters: {model.n_iter_}")
    print(f"  Acc: {acc:.4f}, F1: {f1:.4f}, Dangerous FN: {false_neg:,}")

# Classifier 1 ranking
print(f"\n  Classifier 1 ranking (by F1 / fewest dangerous FN):")
sorted_clf1 = sorted(clf1_results.items(), key=lambda x: -x[1]['f1'])
for rank, (name, r) in enumerate(sorted_clf1, 1):
    print(f"  {rank}. {name}: F1={r['f1']:.4f}, FN={r['false_negatives']:,}, Time={r['train_time_s']:.0f}s")

best_clf1_config = sorted_clf1[0][0]
print(f"  → Best: {best_clf1_config}")

# --- Classifier 2 ---
print(f"\n  === CLASSIFIER 2 (fuel infeasibility) ===")
clf2_results = {}

for config_name, config in CLF_CONFIGS.items():
    print(f"\n  --- {config_name} ---")
    
    t0 = time.time()
    model = MLPClassifier(
        hidden_layer_sizes=config['hidden_layer_sizes'],
        activation='relu', solver='adam',
        learning_rate='adaptive',
        learning_rate_init=config['learning_rate_init'],
        max_iter=300, early_stopping=True, validation_fraction=0.1,
        n_iter_no_change=15,
        batch_size=config['batch_size'],
        random_state=RANDOM_STATE, verbose=False
    )
    model.fit(X_train_clf2, y_clf2_train)
    train_time = time.time() - t0
    
    pred_test = model.predict(X_test_clf2)
    cm = confusion_matrix(y_clf2_test, pred_test)
    f1 = f1_score(y_clf2_test, pred_test)
    acc = accuracy_score(y_clf2_test, pred_test)
    
    false_neg = cm[1, 0]  # True fuel-infeasible, predicted fuel-valid (DANGEROUS)
    
    clf2_results[config_name] = {
        'accuracy': float(acc), 'f1': float(f1),
        'false_negatives': int(false_neg),
        'train_time_s': float(train_time),
        'iterations': int(model.n_iter_)
    }
    
    print(f"  Time: {train_time:.1f}s, Iters: {model.n_iter_}")
    print(f"  Acc: {acc:.4f}, F1: {f1:.4f}, Dangerous FN: {false_neg:,}")

sorted_clf2 = sorted(clf2_results.items(), key=lambda x: -x[1]['f1'])
for rank, (name, r) in enumerate(sorted_clf2, 1):
    print(f"  {rank}. {name}: F1={r['f1']:.4f}, FN={r['false_negatives']:,}, Time={r['train_time_s']:.0f}s")

best_clf2_config = sorted_clf2[0][0]
print(f"  → Best: {best_clf2_config}")

# ============================================================
# STEP 4: PLOTS
# ============================================================
print(f"\n[4/5] Generating plots")

# Plot 1: Regressor comparison - speed MAE
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Speed MAE
ax = axes[0]
configs = [name for name, _ in sorted_reg]
speed_maes = [reg_results[name]['per_output']['speed']['MAE'] for name in configs]
colors = ['tab:green' if name == best_reg_config else 'tab:blue' for name in configs]
bars = ax.barh(configs, speed_maes, color=colors, alpha=0.8)
ax.set_xlabel('MAE (kn)', fontsize=10)
ax.set_title('Ship Speed MAE', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, speed_maes):
    ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', ha='left', va='center', fontsize=8)

# Fuel MAE
ax = axes[1]
fuel_maes = [reg_results[name]['per_output']['fuel']['MAE'] for name in configs]
bars = ax.barh(configs, fuel_maes, color=colors, alpha=0.8)
ax.set_xlabel('MAE (kg/nm)', fontsize=10)
ax.set_title('Fuel Consumption MAE', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
for bar, val in zip(bars, fuel_maes):
    ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', ha='left', va='center', fontsize=8)

# Average R²
ax = axes[2]
avg_r2s = [reg_results[name]['avg_R2'] for name in configs]
bars = ax.barh(configs, avg_r2s, color=colors, alpha=0.8)
ax.set_xlabel('Average R²', fontsize=10)
ax.set_title('Average R² (all 10 outputs)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.set_xlim([min(avg_r2s) * 0.9999, 1.0])
for bar, val in zip(bars, avg_r2s):
    ax.text(bar.get_width(), bar.get_y() + bar.get_height()/2,
            f'{val:.6f}', ha='left', va='center', fontsize=8)

plt.suptitle('Step 5: Regressor Architecture Comparison', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step5_regressor_tuning.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: step5_regressor_tuning.png")

# Plot 2: Per-output comparison of top 3 configs
fig, axes = plt.subplots(2, 5, figsize=(25, 10))
top3_names = [name for name, _ in sorted_reg[:3]]
top3_colors = ['tab:green', 'tab:blue', 'tab:orange']

for i, col in enumerate(OUTPUT_COLS):
    ax = axes[i // 5, i % 5]
    
    maes = [reg_results[name]['per_output'][col]['MAE'] for name in top3_names]
    bars = ax.bar(range(len(top3_names)), maes, color=top3_colors, alpha=0.8)
    
    ax.set_xticks(range(len(top3_names)))
    ax.set_xticklabels([n.split('(')[0].strip() for n in top3_names], fontsize=7, rotation=15)
    ax.set_title(f'{col} [{OUTPUT_UNITS[col]}]', fontsize=10, fontweight='bold')
    ax.set_ylabel('MAE', fontsize=8)
    ax.grid(True, alpha=0.3, axis='y')
    
    for bar, val in zip(bars, maes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{val:.4f}', ha='center', va='bottom', fontsize=7)

plt.suptitle(f'Step 5: Top 3 Regressor Configs — Per Output MAE', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step5_top3_per_output.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: step5_top3_per_output.png")

# Plot 3: Training time vs accuracy trade-off
fig, ax = plt.subplots(figsize=(10, 6))
for name in configs:
    r = reg_results[name]
    ax.scatter(r['train_time_s'], r['per_output']['speed']['MAE'],
               s=100, alpha=0.8, zorder=5)
    ax.annotate(name.split('(')[0].strip(), (r['train_time_s'], r['per_output']['speed']['MAE']),
                textcoords='offset points', xytext=(5, 5), fontsize=7)

ax.set_xlabel('Training Time (seconds)', fontsize=11)
ax.set_ylabel('Speed MAE (kn)', fontsize=11)
ax.set_title('Training Time vs Accuracy Trade-off', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step5_time_vs_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: step5_time_vs_accuracy.png")

# ============================================================
# STEP 5: FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"STEP 5 FINAL SUMMARY")
print(f"{'='*70}")

print(f"\nBest Regressor: {best_reg_config}")
r = reg_results[best_reg_config]
print(f"  Avg MAE: {r['avg_MAE']:.4f}")
print(f"  Avg R²: {r['avg_R2']:.6f}")
print(f"  Speed MAE: {r['per_output']['speed']['MAE']:.4f} kn")
print(f"  Fuel MAE: {r['per_output']['fuel']['MAE']:.4f} kg/nm")
print(f"  Training time: {r['train_time_s']:.0f}s")
print(f"  Iterations: {r['iterations']}")
print(f"\n  Per-output breakdown:")
print(f"  {'Output':<15} {'MAE':<12} {'R²':<14} {'Overest%':<10}")
print(f"  {'-'*51}")
for col in OUTPUT_COLS:
    m = r['per_output'][col]
    print(f"  {col:<15} {m['MAE']:<12.4f} {m['R2']:<14.6f} {m['Overest_pct']:<10.1f}")

print(f"\nBest Classifier 1: {best_clf1_config}")
c1 = clf1_results[best_clf1_config]
print(f"  F1: {c1['f1']:.4f}, Dangerous FN: {c1['false_negatives']:,}")

print(f"\nBest Classifier 2: {best_clf2_config}")
c2 = clf2_results[best_clf2_config]
print(f"  F1: {c2['f1']:.4f}, Dangerous FN: {c2['false_negatives']:,}")

print(f"\nComparison with Step 4 baseline (512-256-128 / 256-128-64):")
print(f"  Step 4 speed MAE: 0.0237 kn")
print(f"  Step 5 speed MAE: {reg_results[best_reg_config]['per_output']['speed']['MAE']:.4f} kn")
print(f"  Step 4 fuel MAE: 0.4302 kg/nm")
print(f"  Step 5 fuel MAE: {reg_results[best_reg_config]['per_output']['fuel']['MAE']:.4f} kg/nm")

# Save results
save_data = {
    'regressor_tuning': reg_results,
    'classifier1_tuning': clf1_results,
    'classifier2_tuning': clf2_results,
    'best_configs': {
        'regressor': best_reg_config,
        'classifier1': best_clf1_config,
        'classifier2': best_clf2_config
    }
}

results_path = os.path.join(OUTPUT_DIR, 'step5_results.json')
with open(results_path, 'w') as f:
    json.dump(save_data, f, indent=2)

print(f"\nResults saved to: {results_path}")
print(f"\nStep 5 complete!")

"""
S175 Surrogate Model — Step 3: Handling -1 Values (Approach B)
===============================================================
Approach B: Two-stage model
  Stage 1: Classifier — predict whether output is -1 or valid
  Stage 2: Regressor — predict actual value (trained ONLY on valid data)

This should preserve Step 2 regression quality while adding -1 detection.

Following Professor Joanna's instruction:
"return -1 whenever the original model returns it"
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
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
import xgboost as xgb

# ============================================================
# CONFIGURATION
# ============================================================
DATA_PATH = '/home/macierz/mohabdal/S175_sample_5pct_stratified.csv'
OUTPUT_DIR = '/home/macierz/mohabdal/S175_experiments'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_FEATURES = ['draft', 'trim', 'rpm', 'PD', 'Hs', 'Tp', 'Chi', 'Vwind', 'Theta_wind']
TARGET = 'speed'
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ============================================================
# STEP 1: LOAD DATA
# ============================================================
print("=" * 70)
print("S175 SURROGATE MODEL — STEP 3: -1 HANDLING (APPROACH B)")
print("Two-stage: Classifier + Regressor")
print("=" * 70)

print(f"\n[1/9] Loading data")
t0 = time.time()
df = pd.read_csv(DATA_PATH, sep=';')
print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

# Create binary label: 1 = valid, 0 = infeasible (-1)
df['is_valid'] = (df[TARGET] != -1).astype(int)

print(f"  Valid (speed != -1): {df['is_valid'].sum():,} ({100*df['is_valid'].mean():.1f}%)")
print(f"  Invalid (speed == -1): {(1-df['is_valid']).sum():,} ({100*(1-df['is_valid'].mean()):.1f}%)")

# ============================================================
# STEP 2: SPLIT DATA (same split for both stages)
# ============================================================
print(f"\n[2/9] Splitting data 80/10/10")

X = df[INPUT_FEATURES].values
y_speed = df[TARGET].values
y_label = df['is_valid'].values  # 1=valid, 0=infeasible

# Stratification by Hs, wind, and validity status
hs_bins = np.digitize(df['Hs'].values, bins=[2.01, 5.01, 8.01])
vwind_bins = np.digitize(df['Vwind'].values, bins=[7.5, 17.5])
strata = hs_bins * 100 + vwind_bins * 10 + y_label

# Split
X_train, X_temp, y_speed_train, y_speed_temp, y_label_train, y_label_temp, strata_train, strata_temp = \
    train_test_split(X, y_speed, y_label, strata, test_size=0.2, 
                     random_state=RANDOM_STATE, stratify=strata)

X_val, X_test, y_speed_val, y_speed_test, y_label_val, y_label_test = \
    train_test_split(X_temp, y_speed_temp, y_label_temp, test_size=0.5,
                     random_state=RANDOM_STATE, stratify=strata_temp)

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"  Train valid: {y_label_train.sum():,} ({100*y_label_train.mean():.1f}%)")
print(f"  Val   valid: {y_label_val.sum():,} ({100*y_label_val.mean():.1f}%)")
print(f"  Test  valid: {y_label_test.sum():,} ({100*y_label_test.mean():.1f}%)")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 3: STAGE 1 — TRAIN CLASSIFIERS (valid vs -1)
# ============================================================
print(f"\n[3/9] STAGE 1: Training classifiers (valid vs -1)")
print(f"  Training on ALL {len(X_train):,} rows")

classifier_results = {}

# --- RF Classifier ---
print(f"\n  Training RF Classifier...")
t0 = time.time()
rf_clf = RandomForestClassifier(
    n_estimators=100, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=RANDOM_STATE
)
rf_clf.fit(X_train, y_label_train)
rf_clf_time = time.time() - t0
print(f"  Done in {rf_clf_time:.1f}s")

rf_clf_pred_val = rf_clf.predict(X_val)
rf_clf_pred_test = rf_clf.predict(X_test)

print(f"  Validation: Acc={accuracy_score(y_label_val, rf_clf_pred_val):.4f}, "
      f"F1={f1_score(y_label_val, rf_clf_pred_val, pos_label=0):.4f}")
print(f"  Test:       Acc={accuracy_score(y_label_test, rf_clf_pred_test):.4f}, "
      f"F1={f1_score(y_label_test, rf_clf_pred_test, pos_label=0):.4f}")

# --- XGBoost Classifier ---
print(f"\n  Training XGBoost Classifier...")
t0 = time.time()
xgb_clf = xgb.XGBClassifier(
    n_estimators=200, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    n_jobs=-1, random_state=RANDOM_STATE, tree_method='hist',
    verbosity=0, eval_metric='logloss'
)
xgb_clf.fit(X_train, y_label_train, eval_set=[(X_val, y_label_val)], verbose=False)
xgb_clf_time = time.time() - t0
print(f"  Done in {xgb_clf_time:.1f}s")

xgb_clf_pred_val = xgb_clf.predict(X_val)
xgb_clf_pred_test = xgb_clf.predict(X_test)

print(f"  Validation: Acc={accuracy_score(y_label_val, xgb_clf_pred_val):.4f}, "
      f"F1={f1_score(y_label_val, xgb_clf_pred_val, pos_label=0):.4f}")
print(f"  Test:       Acc={accuracy_score(y_label_test, xgb_clf_pred_test):.4f}, "
      f"F1={f1_score(y_label_test, xgb_clf_pred_test, pos_label=0):.4f}")

# --- MLP Classifier ---
print(f"\n  Training MLP Classifier...")
t0 = time.time()
mlp_clf = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
    learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=300, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=15, batch_size=1024, random_state=RANDOM_STATE, verbose=False
)
mlp_clf.fit(X_train_scaled, y_label_train)
mlp_clf_time = time.time() - t0
print(f"  Done in {mlp_clf_time:.1f}s (iterations: {mlp_clf.n_iter_})")

mlp_clf_pred_val = mlp_clf.predict(X_val_scaled)
mlp_clf_pred_test = mlp_clf.predict(X_test_scaled)

print(f"  Validation: Acc={accuracy_score(y_label_val, mlp_clf_pred_val):.4f}, "
      f"F1={f1_score(y_label_val, mlp_clf_pred_val, pos_label=0):.4f}")
print(f"  Test:       Acc={accuracy_score(y_label_test, mlp_clf_pred_test):.4f}, "
      f"F1={f1_score(y_label_test, mlp_clf_pred_test, pos_label=0):.4f}")

# Detailed classification report for all
print(f"\n  Detailed classification reports (Test set):")
clf_preds = {'RF': rf_clf_pred_test, 'XGB': xgb_clf_pred_test, 'MLP': mlp_clf_pred_test}
clf_times = {'RF': rf_clf_time, 'XGB': xgb_clf_time, 'MLP': mlp_clf_time}

for name, pred in clf_preds.items():
    print(f"\n  --- {name} Classifier ---")
    cm = confusion_matrix(y_label_test, pred)
    print(f"  Confusion Matrix:")
    print(f"    True -1,    predicted -1:    {cm[0,0]:>8,} (True Negative)")
    print(f"    True -1,    predicted valid: {cm[0,1]:>8,} (False Negative - DANGEROUS)")
    print(f"    True valid, predicted -1:    {cm[1,0]:>8,} (False Positive - conservative)")
    print(f"    True valid, predicted valid: {cm[1,1]:>8,} (True Positive)")
    
    acc = accuracy_score(y_label_test, pred)
    f1_neg = f1_score(y_label_test, pred, pos_label=0)
    prec_neg = precision_score(y_label_test, pred, pos_label=0)
    rec_neg = recall_score(y_label_test, pred, pos_label=0)
    
    classifier_results[name] = {
        'accuracy': float(acc),
        'f1_for_neg1': float(f1_neg),
        'precision_for_neg1': float(prec_neg),
        'recall_for_neg1': float(rec_neg),
        'TN': int(cm[0,0]), 'FN_dangerous': int(cm[0,1]),
        'FP_conservative': int(cm[1,0]), 'TP': int(cm[1,1]),
        'train_time_s': clf_times[name]
    }
    
    print(f"  Accuracy: {acc:.4f}")
    print(f"  -1 Precision: {prec_neg:.4f} (of predicted -1, how many are truly -1)")
    print(f"  -1 Recall: {rec_neg:.4f} (of true -1, how many were caught)")
    print(f"  -1 F1: {f1_neg:.4f}")
    print(f"  FALSE NEGATIVES (dangerous): {cm[0,1]:,} — model says valid but actually -1")

# Feature importance for classifiers
print(f"\n  Feature importance for -1 classification:")
print(f"  {'Feature':<15} {'RF':<10} {'XGB':<10}")
print(f"  {'-'*35}")
rf_imp = rf_clf.feature_importances_
xgb_imp = xgb_clf.feature_importances_
for i, feat in enumerate(INPUT_FEATURES):
    print(f"  {feat:<15} {rf_imp[i]:<10.4f} {xgb_imp[i]:<10.4f}")

# ============================================================
# STEP 4: STAGE 2 — TRAIN REGRESSORS (valid data only)
# ============================================================
print(f"\n[4/9] STAGE 2: Training regressors (valid data ONLY)")

# Filter to valid rows only
valid_train_mask = y_label_train == 1
valid_val_mask = y_label_val == 1
valid_test_mask = y_label_test == 1

X_train_valid = X_train[valid_train_mask]
X_val_valid = X_val[valid_val_mask]
X_test_valid = X_test[valid_test_mask]

y_train_valid = y_speed_train[valid_train_mask]
y_val_valid = y_speed_val[valid_val_mask]
y_test_valid = y_speed_test[valid_test_mask]

X_train_valid_scaled = X_train_scaled[valid_train_mask]
X_val_valid_scaled = X_val_scaled[valid_val_mask]
X_test_valid_scaled = X_test_scaled[valid_test_mask]

print(f"  Valid training rows: {len(X_train_valid):,}")
print(f"  Valid val rows: {len(X_val_valid):,}")
print(f"  Valid test rows: {len(X_test_valid):,}")

regressor_results = {}

# --- RF Regressor ---
print(f"\n  Training RF Regressor (valid data only)...")
t0 = time.time()
rf_reg = RandomForestRegressor(
    n_estimators=100, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=RANDOM_STATE
)
rf_reg.fit(X_train_valid, y_train_valid)
rf_reg_time = time.time() - t0
print(f"  Done in {rf_reg_time:.1f}s")

# --- XGBoost Regressor ---
print(f"\n  Training XGBoost Regressor (valid data only)...")
t0 = time.time()
xgb_reg = xgb.XGBRegressor(
    n_estimators=200, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    n_jobs=-1, random_state=RANDOM_STATE, tree_method='hist', verbosity=0
)
xgb_reg.fit(X_train_valid, y_train_valid, 
            eval_set=[(X_val_valid, y_val_valid)], verbose=False)
xgb_reg_time = time.time() - t0
print(f"  Done in {xgb_reg_time:.1f}s")

# --- MLP Regressor ---
print(f"\n  Training MLP Regressor (valid data only)...")
t0 = time.time()
mlp_reg = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
    learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=300, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=15, batch_size=1024, random_state=RANDOM_STATE, verbose=False
)
mlp_reg.fit(X_train_valid_scaled, y_train_valid)
mlp_reg_time = time.time() - t0
print(f"  Done in {mlp_reg_time:.1f}s (iterations: {mlp_reg.n_iter_})")

reg_times = {'RF': rf_reg_time, 'XGB': xgb_reg_time, 'MLP': mlp_reg_time}
regressors = {'RF': rf_reg, 'XGB': xgb_reg, 'MLP': mlp_reg}

# ============================================================
# STEP 5: EVALUATE COMBINED PIPELINE ON TEST SET
# ============================================================
print(f"\n[5/9] Evaluating combined pipeline (classifier + regressor) on test set")
print(f"  Testing ALL 9 combinations (3 classifiers × 3 regressors)")

combined_results = {}

for clf_name in ['RF', 'XGB', 'MLP']:
    for reg_name in ['RF', 'XGB', 'MLP']:
        combo_name = f"{clf_name}_clf + {reg_name}_reg"
        
        # Stage 1: Classify
        clf_pred = clf_preds[clf_name]  # already computed on test set
        
        # Stage 2: Regress (only for predicted-valid cases)
        pred_valid_mask = (clf_pred == 1)
        
        # Get regression predictions
        if reg_name == 'MLP':
            reg_pred = regressors[reg_name].predict(X_test_scaled[pred_valid_mask])
        else:
            reg_pred = regressors[reg_name].predict(X_test[pred_valid_mask])
        
        # Build final predictions
        final_pred = np.full(len(X_test), -1.0)  # default: -1
        final_pred[pred_valid_mask] = reg_pred
        
        # === CLASSIFICATION METRICS ===
        true_is_neg = (y_label_test == 0)
        pred_is_neg = (clf_pred == 0)
        
        cm = confusion_matrix(true_is_neg, pred_is_neg)
        clf_acc = accuracy_score(true_is_neg, pred_is_neg)
        clf_f1 = f1_score(true_is_neg, pred_is_neg)
        clf_prec = precision_score(true_is_neg, pred_is_neg, zero_division=0)
        clf_rec = recall_score(true_is_neg, pred_is_neg)
        
        # === REGRESSION METRICS (on correctly classified valid rows) ===
        both_truly_valid = (y_label_test == 1) & (clf_pred == 1)
        
        if both_truly_valid.sum() > 0:
            y_true_valid = y_speed_test[both_truly_valid]
            y_pred_valid = final_pred[both_truly_valid]
            
            mae = mean_absolute_error(y_true_valid, y_pred_valid)
            rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
            r2 = r2_score(y_true_valid, y_pred_valid)
            max_err = np.max(np.abs(y_true_valid - y_pred_valid))
            
            errors = y_pred_valid - y_true_valid
            overest_pct = 100 * (errors > 0).sum() / len(errors)
            mean_overest = errors[errors > 0].mean() if (errors > 0).any() else 0
            mean_underest = errors[errors < 0].mean() if (errors < 0).any() else 0
        else:
            mae = rmse = r2 = max_err = overest_pct = mean_overest = mean_underest = float('nan')
        
        # Total inference time estimate
        total_train_time = clf_times[clf_name] + reg_times[reg_name]
        
        combined_results[combo_name] = {
            'classifier': clf_name,
            'regressor': reg_name,
            'total_train_time_s': total_train_time,
            'classification': {
                'accuracy': float(clf_acc), 'f1': float(clf_f1),
                'precision': float(clf_prec), 'recall': float(clf_rec),
                'TN': int(cm[0,0]), 'FP': int(cm[0,1]),
                'FN': int(cm[1,0]), 'TP': int(cm[1,1]),
                'false_negatives_dangerous': int(cm[0,1])
            },
            'regression': {
                'n_rows': int(both_truly_valid.sum()),
                'MAE': float(mae), 'RMSE': float(rmse), 'R2': float(r2),
                'Max_Error': float(max_err), 'Overestimated_pct': float(overest_pct),
                'Mean_overestimation': float(mean_overest),
                'Mean_underestimation': float(mean_underest)
            }
        }

# ============================================================
# STEP 6: RESULTS TABLE
# ============================================================
print(f"\n[6/9] Results — All 9 combinations")

print(f"\n{'Combination':<25} {'Clf F1':<10} {'Reg MAE':<10} {'Reg R²':<14} "
      f"{'MaxErr':<10} {'Overest%':<10} {'FalseNeg':<10} {'TrainTime':<10}")
print(f"{'-'*99}")

for combo_name in sorted(combined_results.keys()):
    r = combined_results[combo_name]
    c = r['classification']
    g = r['regression']
    print(f"{combo_name:<25} {c['f1']:<10.4f} {g['MAE']:<10.4f} {g['R2']:<14.6f} "
          f"{g['Max_Error']:<10.4f} {g['Overestimated_pct']:<10.1f} "
          f"{c['false_negatives_dangerous']:<10,} {r['total_train_time_s']:<10.1f}")

# ============================================================
# STEP 7: FIND BEST COMBINATIONS
# ============================================================
print(f"\n[7/9] Best combinations")

# Best by regression MAE
best_mae = min(combined_results.keys(), key=lambda k: combined_results[k]['regression']['MAE'])
# Best by classification F1
best_f1 = max(combined_results.keys(), key=lambda k: combined_results[k]['classification']['f1'])
# Best by lowest overestimation
best_overest = min(combined_results.keys(), key=lambda k: combined_results[k]['regression']['Overestimated_pct'])
# Best overall (low MAE + high F1 + low false negatives)
best_overall = min(combined_results.keys(), 
    key=lambda k: combined_results[k]['regression']['MAE'] 
                  - combined_results[k]['classification']['f1'] * 0.1
                  + combined_results[k]['classification']['false_negatives_dangerous'] * 0.0001)

print(f"  Best regression (MAE):     {best_mae}")
r = combined_results[best_mae]
print(f"    MAE={r['regression']['MAE']:.4f}, R²={r['regression']['R2']:.6f}, "
      f"F1={r['classification']['f1']:.4f}")

print(f"  Best -1 detection (F1):    {best_f1}")
r = combined_results[best_f1]
print(f"    F1={r['classification']['f1']:.4f}, MAE={r['regression']['MAE']:.4f}")

print(f"  Lowest overestimation:     {best_overest}")
r = combined_results[best_overest]
print(f"    Overest={r['regression']['Overestimated_pct']:.1f}%, MAE={r['regression']['MAE']:.4f}")

print(f"  Best overall:              {best_overall}")
r = combined_results[best_overall]
print(f"    MAE={r['regression']['MAE']:.4f}, F1={r['classification']['f1']:.4f}, "
      f"Overest={r['regression']['Overestimated_pct']:.1f}%")

# ============================================================
# STEP 8: COMPARISON WITH ALL APPROACHES
# ============================================================
print(f"\n[8/9] Comparison across ALL approaches")

# Load previous results
step2_path = os.path.join(OUTPUT_DIR, 'step2_results.json')
approach_a_path = os.path.join(OUTPUT_DIR, 'step3_approach_a_results.json')

step2 = {}
approach_a = {}
if os.path.exists(step2_path):
    with open(step2_path, 'r') as f:
        step2 = json.load(f)
if os.path.exists(approach_a_path):
    with open(approach_a_path, 'r') as f:
        approach_a = json.load(f)

print(f"\n{'Approach':<30} {'MAE(kn)':<10} {'R²':<14} {'-1 F1':<10} {'MaxErr':<10} {'Overest%':<10}")
print(f"{'-'*84}")

# Step 2
if 'MLP' in step2:
    print(f"{'Step2 MLP (no -1)':<30} {step2['MLP']['MAE']:<10.4f} "
          f"{step2['MLP']['R2']:<14.6f} {'N/A':<10} {step2['MLP']['Max_Error']:<10.4f} "
          f"{step2['MLP']['Overestimated_pct']:<10.1f}")

# Approach A
if 'MLP' in approach_a:
    a = approach_a['MLP']
    print(f"{'Approach A (MLP, enc=-1)':<30} {a['regression_valid']['MAE']:<10.4f} "
          f"{a['regression_valid']['R2']:<14.6f} {a['classification']['f1']:<10.4f} "
          f"{a['regression_valid']['Max_Error']:<10.4f} "
          f"{a['regression_valid']['Overestimated_pct']:<10.1f}")

# Approach B best
r = combined_results[best_overall]
print(f"{'Approach B: '+best_overall:<30} {r['regression']['MAE']:<10.4f} "
      f"{r['regression']['R2']:<14.6f} {r['classification']['f1']:<10.4f} "
      f"{r['regression']['Max_Error']:<10.4f} "
      f"{r['regression']['Overestimated_pct']:<10.1f}")

# ============================================================
# STEP 9: PLOTS
# ============================================================
print(f"\n[9/9] Generating plots")

# Plot 1: Confusion matrices for all 3 classifiers
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, name in enumerate(['RF', 'XGB', 'MLP']):
    ax = axes[idx]
    r = classifier_results[name]
    cm = np.array([[r['TN'], r['FN_dangerous']], [r['FP_conservative'], r['TP']]])
    
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center',
                    color=color, fontsize=11, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Pred -1', 'Pred Valid'], fontsize=9)
    ax.set_yticklabels(['True -1', 'True Valid'], fontsize=9)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    ax.set_title(f'{name} Classifier\nF1={r["f1_for_neg1"]:.4f}, '
                 f'Dangerous FN={r["FN_dangerous"]:,}', fontsize=10)

plt.suptitle('Approach B — Stage 1: Classifier Performance (Test Set)', 
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step3_approach_b_classifiers.png'), 
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: step3_approach_b_classifiers.png")

# Plot 2: Regression comparison (best combo vs Step 2)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Get the best regressor predictions on valid test data
for idx, reg_name in enumerate(['RF', 'XGB', 'MLP']):
    ax = axes[idx]
    
    if reg_name == 'MLP':
        reg_pred = regressors[reg_name].predict(X_test_valid_scaled)
    else:
        reg_pred = regressors[reg_name].predict(X_test_valid)
    
    mae = mean_absolute_error(y_test_valid, reg_pred)
    r2 = r2_score(y_test_valid, reg_pred)
    
    n_plot = min(20000, len(y_test_valid))
    plot_idx = np.random.choice(len(y_test_valid), n_plot, replace=False)
    
    ax.scatter(y_test_valid[plot_idx], reg_pred[plot_idx], alpha=0.2, s=5, c='tab:blue')
    
    min_v = y_test_valid.min()
    max_v = y_test_valid.max()
    ax.plot([min_v, max_v], [min_v, max_v], 'r--', linewidth=1.5, label='Perfect')
    
    ax.set_xlabel('Actual Speed (kn)', fontsize=10)
    ax.set_ylabel('Predicted Speed (kn)', fontsize=10)
    ax.set_title(f'{reg_name} Regressor (valid only)\n'
                 f'MAE={mae:.4f}, R²={r2:.6f}', fontsize=11)
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

plt.suptitle('Approach B — Stage 2: Regressor Performance (Valid Test Data Only)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step3_approach_b_regressors.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: step3_approach_b_regressors.png")

# Plot 3: Approach comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

approaches = ['Step 2\n(no -1)', 'Approach A\n(enc=-1)', f'Approach B\n({best_overall})']
colors = ['tab:blue', 'tab:orange', 'tab:green']

# MAE comparison
ax = axes[0]
mae_vals = [
    step2.get('MLP', {}).get('MAE', 0),
    approach_a.get('MLP', {}).get('regression_valid', {}).get('MAE', 0),
    combined_results[best_overall]['regression']['MAE']
]
ax.bar(approaches, mae_vals, color=colors, alpha=0.8)
ax.set_ylabel('MAE (kn)', fontsize=11)
ax.set_title('Regression MAE\n(lower is better)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(mae_vals):
    ax.text(i, v + 0.002, f'{v:.4f}', ha='center', fontsize=9, fontweight='bold')

# R² comparison
ax = axes[1]
r2_vals = [
    step2.get('MLP', {}).get('R2', 0),
    approach_a.get('MLP', {}).get('regression_valid', {}).get('R2', 0),
    combined_results[best_overall]['regression']['R2']
]
ax.bar(approaches, r2_vals, color=colors, alpha=0.8)
ax.set_ylabel('R²', fontsize=11)
ax.set_title('Regression R²\n(higher is better)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(r2_vals):
    ax.text(i, v + 0.005, f'{v:.6f}', ha='center', fontsize=9, fontweight='bold')

# F1 comparison
ax = axes[2]
f1_vals = [
    0,  # Step 2 cannot detect -1
    approach_a.get('MLP', {}).get('classification', {}).get('f1', 0),
    combined_results[best_overall]['classification']['f1']
]
ax.bar(approaches, f1_vals, color=colors, alpha=0.8)
ax.set_ylabel('F1 Score', fontsize=11)
ax.set_title('-1 Detection F1\n(higher is better)', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, v in enumerate(f1_vals):
    label = 'N/A' if v == 0 else f'{v:.4f}'
    ax.text(i, v + 0.01, label, ha='center', fontsize=9, fontweight='bold')

plt.suptitle('Comparison: Step 2 vs Approach A vs Approach B', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step3_approach_b_comparison.png'),
            dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: step3_approach_b_comparison.png")

# ============================================================
# SAVE RESULTS
# ============================================================
results_path = os.path.join(OUTPUT_DIR, 'step3_approach_b_results.json')
save_data = {
    'classifier_results': classifier_results,
    'combined_results': combined_results,
    'best_combinations': {
        'best_mae': best_mae,
        'best_f1': best_f1,
        'best_overest': best_overest,
        'best_overall': best_overall
    }
}
with open(results_path, 'w') as f:
    json.dump(save_data, f, indent=2)

print(f"\nResults saved to: {results_path}")
print(f"Plots saved to: {OUTPUT_DIR}/")
print(f"\nStep 3 (Approach B) complete!")

"""
S175 Surrogate Model — Step 3: Handling -1 Values (Approach A)
===============================================================
Approach A: Include -1 as a valid target value in regression.
Train models on ALL data (including -1 rows).
After prediction, apply threshold to classify -1 vs valid.

Test with ship speed first (same as Step 2 but now including -1 rows).

Following Professor Joanna's instruction:
"return -1 whenever the original model returns it"
"try Approach A first, then C, then B"
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
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
# STEP 1: LOAD ALL DATA (including -1 rows)
# ============================================================
print("=" * 70)
print("S175 SURROGATE MODEL — STEP 3: -1 HANDLING (APPROACH A)")
print("=" * 70)

print(f"\n[1/8] Loading ALL data (including -1 rows)")
t0 = time.time()
df = pd.read_csv(DATA_PATH, sep=';')
print(f"  Loaded {len(df):,} rows in {time.time()-t0:.1f}s")

# Count row categories
all_neg = (df[['speed','power','torque','lat_acc','MSI',
               'roll','slam','green_water','prop_emerg','fuel']] == -1).all(axis=1)
fuel_neg = (df['fuel'] == -1) & ~all_neg
valid = ~all_neg & ~fuel_neg

print(f"\n  Row categories:")
print(f"    Fully valid:    {valid.sum():>10,} ({100*valid.sum()/len(df):.1f}%)")
print(f"    Partial -1:     {fuel_neg.sum():>10,} ({100*fuel_neg.sum()/len(df):.1f}%)")
print(f"    All -1:         {all_neg.sum():>10,} ({100*all_neg.sum()/len(df):.1f}%)")

# For ship speed specifically:
speed_valid = (df[TARGET] != -1).sum()
speed_neg = (df[TARGET] == -1).sum()
print(f"\n  Ship speed specifically:")
print(f"    Valid (speed != -1): {speed_valid:,} ({100*speed_valid/len(df):.1f}%)")
print(f"    Invalid (speed == -1): {speed_neg:,} ({100*speed_neg/len(df):.1f}%)")
print(f"    Valid speed range: [{df.loc[df[TARGET]!=-1, TARGET].min():.4f}, "
      f"{df.loc[df[TARGET]!=-1, TARGET].max():.4f}]")

# ============================================================
# STEP 2: PREPARE DATA — USE ALL ROWS
# ============================================================
print(f"\n[2/8] Preparing data (ALL rows, -1 included as target)")

X = df[INPUT_FEATURES].values
y = df[TARGET].values  # includes -1 values

# Create stratification key for splitting
# Need to ensure both valid and -1 cases are in all splits
hs_bins = np.digitize(df['Hs'].values, bins=[2.01, 5.01, 8.01])
vwind_bins = np.digitize(df['Vwind'].values, bins=[7.5, 17.5])
speed_status = (y == -1).astype(int)  # 0=valid, 1=invalid
strata = hs_bins * 100 + vwind_bins * 10 + speed_status

# Split 80/10/10
X_train, X_temp, y_train, y_temp, strata_train, strata_temp = train_test_split(
    X, y, strata, test_size=0.2, random_state=RANDOM_STATE, stratify=strata
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=RANDOM_STATE, stratify=strata_temp
)

print(f"  Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")
print(f"\n  Train -1 count: {(y_train==-1).sum():,} ({100*(y_train==-1).mean():.1f}%)")
print(f"  Val   -1 count: {(y_val==-1).sum():,} ({100*(y_val==-1).mean():.1f}%)")
print(f"  Test  -1 count: {(y_test==-1).sum():,} ({100*(y_test==-1).mean():.1f}%)")

# Scale features for MLP
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ============================================================
# STEP 3: TRAIN ALL 3 MODELS ON DATA WITH -1
# ============================================================
print(f"\n[3/8] Training models on ALL data (including -1 as target)")

models = {}
predictions = {}
train_times = {}

# --- Random Forest ---
print(f"\n  Training Random Forest...")
t0 = time.time()
rf = RandomForestRegressor(
    n_estimators=100, max_depth=20, min_samples_leaf=5,
    n_jobs=-1, random_state=RANDOM_STATE
)
rf.fit(X_train, y_train)
train_times['RF'] = time.time() - t0
print(f"  Done in {train_times['RF']:.1f}s")

predictions['RF'] = {
    'val': rf.predict(X_val),
    'test': rf.predict(X_test)
}

# --- XGBoost ---
print(f"\n  Training XGBoost...")
t0 = time.time()
xgb_model = xgb.XGBRegressor(
    n_estimators=200, max_depth=8, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
    n_jobs=-1, random_state=RANDOM_STATE, tree_method='hist', verbosity=0
)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
train_times['XGB'] = time.time() - t0
print(f"  Done in {train_times['XGB']:.1f}s")

predictions['XGB'] = {
    'val': xgb_model.predict(X_val),
    'test': xgb_model.predict(X_test)
}

# --- MLP ---
print(f"\n  Training MLP...")
t0 = time.time()
mlp = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64), activation='relu', solver='adam',
    learning_rate='adaptive', learning_rate_init=0.001,
    max_iter=300, early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=15, batch_size=1024, random_state=RANDOM_STATE, verbose=False
)
mlp.fit(X_train_scaled, y_train)
train_times['MLP'] = time.time() - t0
print(f"  Done in {train_times['MLP']:.1f}s (iterations: {mlp.n_iter_})")

predictions['MLP'] = {
    'val': mlp.predict(X_val_scaled),
    'test': mlp.predict(X_test_scaled)
}

# ============================================================
# STEP 4: ANALYZE RAW PREDICTIONS — WHAT DOES THE MODEL PREDICT FOR -1 CASES?
# ============================================================
print(f"\n[4/8] Analyzing raw predictions for -1 cases")
print(f"  (What values do the models predict when the true target is -1?)")

for name in ['RF', 'XGB', 'MLP']:
    pred = predictions[name]['test']
    
    # Predictions where true target is -1
    pred_for_neg = pred[y_test == -1]
    # Predictions where true target is valid
    pred_for_valid = pred[y_test != -1]
    
    print(f"\n  --- {name} ---")
    print(f"  When true = -1 (should predict near -1):")
    print(f"    min={pred_for_neg.min():.4f}, mean={pred_for_neg.mean():.4f}, "
          f"max={pred_for_neg.max():.4f}, std={pred_for_neg.std():.4f}")
    print(f"  When true = valid (should predict positive values):")
    print(f"    min={pred_for_valid.min():.4f}, mean={pred_for_valid.mean():.4f}, "
          f"max={pred_for_valid.max():.4f}")
    
    # Distribution of predictions for -1 cases
    print(f"\n  Prediction distribution for true=-1 cases:")
    thresholds = [-0.5, 0, 0.5, 1.0, 2.0, 3.0, 5.0]
    for t in thresholds:
        below = (pred_for_neg <= t).sum()
        print(f"    pred <= {t:>4.1f}: {below:>8,} ({100*below/len(pred_for_neg):.1f}%)")

# ============================================================
# STEP 5: FIND OPTIMAL THRESHOLD
# ============================================================
print(f"\n[5/8] Finding optimal threshold for each model")
print(f"  (Threshold: if prediction <= threshold, classify as -1)")

best_thresholds = {}

for name in ['RF', 'XGB', 'MLP']:
    pred_val = predictions[name]['val']
    
    # Try different thresholds on validation set
    thresholds = np.arange(-2.0, 5.1, 0.1)
    best_f1 = -1
    best_t = 0
    
    true_is_neg = (y_val == -1)
    
    results_per_threshold = []
    for t in thresholds:
        pred_is_neg = (pred_val <= t)
        
        # Metrics
        acc = accuracy_score(true_is_neg, pred_is_neg)
        f1 = f1_score(true_is_neg, pred_is_neg)
        prec = precision_score(true_is_neg, pred_is_neg, zero_division=0)
        rec = recall_score(true_is_neg, pred_is_neg)
        
        results_per_threshold.append({
            'threshold': t, 'accuracy': acc, 'f1': f1, 
            'precision': prec, 'recall': rec
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
    
    best_thresholds[name] = best_t
    
    print(f"\n  --- {name} ---")
    print(f"  Best threshold: {best_t:.1f}")
    print(f"  Best F1-score: {best_f1:.4f}")
    
    # Show metrics at best threshold
    pred_is_neg = (pred_val <= best_t)
    cm = confusion_matrix(true_is_neg, pred_is_neg)
    print(f"  Confusion matrix (val set):")
    print(f"    True valid,  predicted valid:  {cm[0,0]:>8,} (True Negative)")
    print(f"    True valid,  predicted -1:     {cm[0,1]:>8,} (False Positive)")
    print(f"    True -1,     predicted valid:  {cm[1,0]:>8,} (False Negative)")
    print(f"    True -1,     predicted -1:     {cm[1,1]:>8,} (True Positive)")
    print(f"  Accuracy: {accuracy_score(true_is_neg, pred_is_neg):.4f}")
    print(f"  Precision: {precision_score(true_is_neg, pred_is_neg):.4f}")
    print(f"  Recall: {recall_score(true_is_neg, pred_is_neg):.4f}")

# ============================================================
# STEP 6: EVALUATE ON TEST SET WITH OPTIMAL THRESHOLD
# ============================================================
print(f"\n[6/8] Evaluating on test set with optimal thresholds")

all_results = {}

for name in ['RF', 'XGB', 'MLP']:
    pred_test = predictions[name]['test']
    threshold = best_thresholds[name]
    
    # Apply threshold
    pred_classified = pred_test.copy()
    pred_classified[pred_test <= threshold] = -1
    
    # -1 Classification metrics
    true_is_neg = (y_test == -1)
    pred_is_neg = (pred_classified == -1)
    
    cm = confusion_matrix(true_is_neg, pred_is_neg)
    acc = accuracy_score(true_is_neg, pred_is_neg)
    f1 = f1_score(true_is_neg, pred_is_neg)
    prec = precision_score(true_is_neg, pred_is_neg)
    rec = recall_score(true_is_neg, pred_is_neg)
    
    # Regression metrics (only on cases where both true and predicted are valid)
    both_valid = (~true_is_neg) & (~pred_is_neg)
    if both_valid.sum() > 0:
        mae_valid = mean_absolute_error(y_test[both_valid], pred_classified[both_valid])
        rmse_valid = np.sqrt(mean_squared_error(y_test[both_valid], pred_classified[both_valid]))
        r2_valid = r2_score(y_test[both_valid], pred_classified[both_valid])
        max_err_valid = np.max(np.abs(y_test[both_valid] - pred_classified[both_valid]))
        
        # Overestimation on valid predictions
        errors_valid = pred_classified[both_valid] - y_test[both_valid]
        overest_pct = 100 * (errors_valid > 0).sum() / len(errors_valid)
    else:
        mae_valid = rmse_valid = r2_valid = max_err_valid = overest_pct = float('nan')
    
    print(f"\n  === {name} (threshold={threshold:.1f}) ===")
    print(f"  -1 Classification (test set):")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall:    {rec:.4f}")
    print(f"    F1-score:  {f1:.4f}")
    print(f"    Confusion: TN={cm[0,0]:,} FP={cm[0,1]:,} FN={cm[1,0]:,} TP={cm[1,1]:,}")
    print(f"  Regression on valid predictions:")
    print(f"    Rows: {both_valid.sum():,}")
    print(f"    MAE:  {mae_valid:.4f} kn")
    print(f"    RMSE: {rmse_valid:.4f} kn")
    print(f"    R²:   {r2_valid:.6f}")
    print(f"    Max Error: {max_err_valid:.4f} kn")
    print(f"    Overestimation: {overest_pct:.1f}%")
    
    all_results[name] = {
        'threshold': threshold,
        'classification': {
            'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1,
            'TN': int(cm[0,0]), 'FP': int(cm[0,1]),
            'FN': int(cm[1,0]), 'TP': int(cm[1,1])
        },
        'regression_valid': {
            'n_rows': int(both_valid.sum()),
            'MAE': mae_valid, 'RMSE': rmse_valid, 'R2': r2_valid,
            'Max_Error': max_err_valid, 'Overestimated_pct': overest_pct
        },
        'train_time_s': train_times[name]
    }

# ============================================================
# STEP 7: COMPARISON WITH STEP 2 (valid-only training)
# ============================================================
print(f"\n[7/8] Comparison: Step 2 (valid only) vs Step 3 (with -1)")

# Load Step 2 results
step2_path = os.path.join(OUTPUT_DIR, 'step2_results.json')
if os.path.exists(step2_path):
    with open(step2_path, 'r') as f:
        step2_results = json.load(f)
    
    print(f"\n  {'Model':<8} {'Metric':<12} {'Step2 (valid only)':<22} {'Step3 Approach A':<22} {'Difference':<15}")
    print(f"  {'-'*79}")
    
    name_map = {'RF': 'RandomForest', 'XGB': 'XGBoost', 'MLP': 'MLP'}
    for name in ['RF', 'XGB', 'MLP']:
        s2_name = name_map[name]
        if s2_name in step2_results:
            s2_mae = step2_results[s2_name]['MAE']
            s3_mae = all_results[name]['regression_valid']['MAE']
            s2_r2 = step2_results[s2_name]['R2']
            s3_r2 = all_results[name]['regression_valid']['R2']
            
            print(f"  {name:<8} {'MAE (kn)':<12} {s2_mae:<22.4f} {s3_mae:<22.4f} {s3_mae-s2_mae:<+15.4f}")
            print(f"  {name:<8} {'R²':<12} {s2_r2:<22.6f} {s3_r2:<22.6f} {s3_r2-s2_r2:<+15.6f}")
            print(f"  {'':<8} {'+ -1 F1':<12} {'N/A':<22} {all_results[name]['classification']['f1']:<22.4f}")
            print(f"  {'-'*79}")
else:
    print(f"  Step 2 results not found at {step2_path}")

# ============================================================
# STEP 8: PLOTS
# ============================================================
print(f"\n[8/8] Generating plots")

# Plot 1: Raw prediction distribution for -1 cases
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, name in enumerate(['RF', 'XGB', 'MLP']):
    ax = axes[idx]
    pred_test = predictions[name]['test']
    threshold = best_thresholds[name]
    
    # Predictions for true=-1 cases
    pred_neg = pred_test[y_test == -1]
    # Predictions for true=valid cases
    pred_valid = pred_test[y_test != -1]
    
    ax.hist(pred_neg, bins=100, alpha=0.7, color='red', label='True = -1', density=True)
    ax.hist(pred_valid, bins=100, alpha=0.5, color='blue', label='True = valid', density=True)
    ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2, 
               label=f'Threshold={threshold:.1f}')
    ax.axvline(x=-1, color='green', linestyle=':', linewidth=1.5, label='Exact -1')
    
    ax.set_xlabel('Predicted Value', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title(f'{name} — Prediction Distribution\n'
                 f'F1={all_results[name]["classification"]["f1"]:.4f}', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.suptitle('Step 3: Approach A — How Models Predict -1 Cases', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step3_approach_a_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: step3_approach_a_distribution.png")

# Plot 2: Predicted vs Actual (with -1 cases highlighted)
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for idx, name in enumerate(['RF', 'XGB', 'MLP']):
    ax = axes[idx]
    pred_test = predictions[name]['test']
    threshold = best_thresholds[name]
    
    # Subsample for plotting
    n_plot = min(20000, len(y_test))
    plot_idx = np.random.choice(len(y_test), n_plot, replace=False)
    
    # Separate valid and -1 cases
    valid_mask = y_test[plot_idx] != -1
    neg_mask = y_test[plot_idx] == -1
    
    ax.scatter(y_test[plot_idx][valid_mask], pred_test[plot_idx][valid_mask], 
               alpha=0.2, s=5, c='blue', label='Valid cases')
    ax.scatter(y_test[plot_idx][neg_mask], pred_test[plot_idx][neg_mask], 
               alpha=0.3, s=10, c='red', label='-1 cases')
    
    # Perfect line
    min_v = min(y_test.min(), pred_test.min())
    max_v = max(y_test.max(), pred_test.max())
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', linewidth=1, label='Perfect')
    
    # Threshold line
    ax.axhline(y=threshold, color='green', linestyle='--', linewidth=1, 
               label=f'Threshold={threshold:.1f}')
    
    ax.set_xlabel('Actual', fontsize=10)
    ax.set_ylabel('Predicted', fontsize=10)
    ax.set_title(f'{name}', fontsize=11)
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

plt.suptitle('Step 3: Approach A — Predicted vs Actual (with -1 cases)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step3_approach_a_scatter.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: step3_approach_a_scatter.png")

# Plot 3: Confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for idx, name in enumerate(['RF', 'XGB', 'MLP']):
    ax = axes[idx]
    r = all_results[name]['classification']
    cm = np.array([[r['TN'], r['FP']], [r['FN'], r['TP']]])
    
    im = ax.imshow(cm, cmap='Blues', interpolation='nearest')
    
    # Add text
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i,j] > cm.max()/2 else 'black'
            ax.text(j, i, f'{cm[i,j]:,}', ha='center', va='center', 
                    color=color, fontsize=11, fontweight='bold')
    
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Valid', '-1'], fontsize=10)
    ax.set_yticklabels(['Valid', '-1'], fontsize=10)
    ax.set_xlabel('Predicted', fontsize=10)
    ax.set_ylabel('Actual', fontsize=10)
    ax.set_title(f'{name}\nF1={r["f1"]:.4f}, Acc={r["accuracy"]:.4f}', fontsize=11)

plt.suptitle('Step 3: Approach A — Confusion Matrices (Test Set)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'step3_approach_a_confusion.png'), dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: step3_approach_a_confusion.png")

# ============================================================
# FINAL SUMMARY
# ============================================================
print(f"\n{'='*70}")
print(f"STEP 3 RESULTS — Approach A: -1 as Target Value (Test Set)")
print(f"{'='*70}")
print(f"\nTarget: ship speed")
print(f"Total test rows: {len(y_test):,}")
print(f"  Valid: {(y_test!=-1).sum():,} | -1: {(y_test==-1).sum():,}")

print(f"\n-1 Classification Performance:")
print(f"{'Model':<8} {'Threshold':<12} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
print(f"{'-'*68}")
for name in ['RF', 'XGB', 'MLP']:
    r = all_results[name]
    c = r['classification']
    print(f"{name:<8} {r['threshold']:<12.1f} {c['accuracy']:<12.4f} "
          f"{c['precision']:<12.4f} {c['recall']:<12.4f} {c['f1']:<12.4f}")

print(f"\nRegression Performance (on correctly classified valid rows):")
print(f"{'Model':<8} {'MAE(kn)':<12} {'RMSE(kn)':<12} {'R²':<14} {'MaxErr(kn)':<12} {'Overest%':<10}")
print(f"{'-'*68}")
for name in ['RF', 'XGB', 'MLP']:
    rv = all_results[name]['regression_valid']
    print(f"{name:<8} {rv['MAE']:<12.4f} {rv['RMSE']:<12.4f} {rv['R2']:<14.6f} "
          f"{rv['Max_Error']:<12.4f} {rv['Overestimated_pct']:<10.1f}")

print(f"\nConclusion for Approach A:")
best_f1_model = max(all_results.keys(), key=lambda k: all_results[k]['classification']['f1'])
best_mae_model = min(all_results.keys(), key=lambda k: all_results[k]['regression_valid']['MAE'])
print(f"  Best -1 detection (F1): {best_f1_model}")
print(f"  Best regression accuracy (MAE): {best_mae_model}")

# Save results
results_path = os.path.join(OUTPUT_DIR, 'step3_approach_a_results.json')

# Convert numpy types to native Python for JSON serialization
def convert_to_native(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    return obj

with open(results_path, 'w') as f:
    json.dump(convert_to_native(all_results), f, indent=2)
print(f"\nResults saved to: {results_path}")
print(f"Plots saved to: {OUTPUT_DIR}/")
print(f"\nStep 3 (Approach A) complete!")

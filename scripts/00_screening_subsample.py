"""
Create a properly stratified 5% subsample from S175_shaft_gen_off.txt
Ensures ALL combinations of (Hs_group, Chi, Wind_speed, Status) are represented
"""
import pandas as pd
import numpy as np
import time

print("=" * 60)
print("STRATIFIED 5% SUBSAMPLE CREATION")
print("=" * 60)

# Column names
cols = [
    'draft','trim','GMt','rpm','PD','shaft_gen',
    'Hs','Tp','Chi','Vwind','Theta_wind',
    'speed','power','torque','lat_acc','MSI',
    'roll','slam','green_water','prop_emerg','fuel'
]

print("\nLoading full dataset (this may take a few minutes)...")
t0 = time.time()
df = pd.read_csv(
    '/home/macierz/mohabdal/S175_shaft_gen_off.txt',
    sep=';', skiprows=4, header=None, names=cols
)
print(f"Loaded {len(df):,} rows in {time.time()-t0:.1f}s")
print(f"Memory usage: {df.memory_usage(deep=True).sum()/1e9:.1f} GB")

# Create stratification key
print("\nCreating stratification groups...")

# Hs groups
df['Hs_group'] = pd.cut(df['Hs'], 
    bins=[-0.01, 2.0, 5.0, 8.0, 10.01],
    labels=['calm', 'moderate', 'rough', 'extreme'])

# Status
all_neg = (df[['speed','power','torque','lat_acc','MSI',
               'roll','slam','green_water','prop_emerg','fuel']] == -1).all(axis=1)
fuel_neg = (df['fuel'] == -1) & ~all_neg
df['status'] = 'valid'
df.loc[fuel_neg, 'status'] = 'partial'
df.loc[all_neg, 'status'] = 'all_neg'

# Combined stratification key (Hs_group + Chi + Wind + Status)
df['strata'] = (df['Hs_group'].astype(str) + '_' + 
                df['Chi'].astype(str) + '_' + 
                df['Vwind'].astype(str) + '_' + 
                df['status'])

# Count strata
n_strata = df['strata'].nunique()
print(f"Total unique strata: {n_strata:,}")
print(f"Smallest stratum: {df['strata'].value_counts().min()} rows")
print(f"Largest stratum: {df['strata'].value_counts().max()} rows")

# Stratified sampling: 5% from each stratum, minimum 50 rows per stratum
print("\nPerforming stratified sampling (5%, min 50 per stratum)...")
t0 = time.time()

def sample_stratum(group):
    n = max(50, int(len(group) * 0.05))
    n = min(n, len(group))  # don't sample more than available
    return group.sample(n=n, random_state=42)

sample = df.groupby('strata', group_keys=False).apply(sample_stratum)
print(f"Sampling done in {time.time()-t0:.1f}s")

# Drop helper columns
sample = sample.drop(columns=['Hs_group', 'status', 'strata'])

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Full dataset:    {len(df):>15,} rows")
print(f"Sample:          {len(sample):>15,} rows ({100*len(sample)/len(df):.2f}%)")

# Verify coverage
print(f"\n--- Coverage verification ---")
for col in ['draft','trim','rpm','PD','Hs','Tp','Chi','Vwind','Theta_wind']:
    full_vals = set(df[col].unique())
    sample_vals = set(sample[col].unique())
    missing = full_vals - sample_vals
    status = "✓ ALL covered" if len(missing)==0 else f"✗ MISSING {len(missing)} values"
    print(f"  {col}: {len(sample_vals)}/{len(full_vals)} values — {status}")

# Verify status distribution
print(f"\n--- Status distribution ---")
full_status = df['status'] if 'status' in df.columns else None
# Recalculate for sample
all_neg_s = (sample[['speed','power','torque','lat_acc','MSI',
                      'roll','slam','green_water','prop_emerg','fuel']] == -1).all(axis=1)
fuel_neg_s = (sample['fuel'] == -1) & ~all_neg_s
valid_s = ~all_neg_s & ~fuel_neg_s

print(f"  Valid:      {valid_s.sum():>10,} ({100*valid_s.sum()/len(sample):.1f}%)")
print(f"  Partial -1: {fuel_neg_s.sum():>10,} ({100*fuel_neg_s.sum()/len(sample):.1f}%)")
print(f"  All -1:     {all_neg_s.sum():>10,} ({100*all_neg_s.sum()/len(sample):.1f}%)")

# Save
print("\nSaving stratified sample...")
sample.to_csv('/home/macierz/mohabdal/S175_sample_5pct_stratified.csv', 
              sep=';', index=False, header=False)
print(f"Saved to ~/S175_sample_5pct_stratified.csv")
print(f"File size: {sample.memory_usage(deep=True).sum()/1e6:.0f} MB (in memory)")

print("\nDone!")

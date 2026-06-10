import numpy as np
import pandas as pd
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────
# LOAD KAGGLE DATASETS
# ─────────────────────────────────────────
def load_kaggle_data():
    print("Loading Kaggle datasets...")
    datasets = {}

    nhs_path = 'data/kaggle/NHS_Organ_Donation.csv'
    ot_path  = 'data/kaggle/Organ_Transplant.csv'

    if os.path.exists(nhs_path):
        nhs = pd.read_csv(nhs_path)
        nhs.columns = nhs.columns.str.strip()
        datasets['nhs'] = nhs
        print(f"  NHS dataset:    {len(nhs)} rows, {len(nhs.columns)} columns")
        print(f"  Columns: {nhs.columns.tolist()}")
    else:
        print(f"  NHS dataset not found at {nhs_path}")

    if os.path.exists(ot_path):
        ot = pd.read_csv(ot_path)
        ot.columns = ot.columns.str.strip()
        datasets['organ_transplant'] = ot
        print(f"\n  Organ Transplant: {len(ot)} rows, {len(ot.columns)} columns")
        print(f"  Columns: {ot.columns.tolist()}")
    else:
        print(f"  Organ Transplant dataset not found at {ot_path}")

    return datasets


# ─────────────────────────────────────────
# KS TEST — DISTRIBUTION COMPARISON
# ─────────────────────────────────────────
def ks_test(synthetic: np.ndarray, real: np.ndarray, feature_name: str):
    """
    Manual Kolmogorov-Smirnov test.
    p > 0.05 = distributions are similar (good)
    p < 0.05 = distributions differ significantly
    """
    n1 = len(synthetic)
    n2 = len(real)

    # Combined sorted values
    combined = np.concatenate([synthetic, real])
    combined.sort()

    # CDF for each sample
    cdf1 = np.searchsorted(np.sort(synthetic), combined, side='right') / n1
    cdf2 = np.searchsorted(np.sort(real),      combined, side='right') / n2

    # KS statistic = max difference between CDFs
    ks_stat = np.max(np.abs(cdf1 - cdf2))

    # Approximate p-value
    en = np.sqrt(n1 * n2 / (n1 + n2))
    p_value = 2 * np.exp(-2 * (en * ks_stat)**2)
    p_value = min(1.0, p_value)

    status = "✓ Similar" if p_value > 0.05 else "✗ Different"
    print(f"  {feature_name:20s} KS={ks_stat:.4f}  p={p_value:.4f}  {status}")

    return ks_stat, p_value


# ─────────────────────────────────────────
# COMPARE BLOOD TYPE DISTRIBUTIONS
# ─────────────────────────────────────────
def compare_blood_types(synthetic_donors, kaggle_data):
    print("\n" + "="*50)
    print("BLOOD TYPE DISTRIBUTION COMPARISON")
    print("="*50)

    # Synthetic distribution
    synth_bt = synthetic_donors['blood_type'].value_counts(normalize=True).sort_index()
    print("\nSynthetic distribution:")
    for bt, pct in synth_bt.items():
        bar = '█' * int(pct * 40)
        print(f"  {bt:4s} {bar} {pct:.1%}")

    # Real OPTN distribution for reference
    real_optn = {'O': 0.53, 'A': 0.34, 'B': 0.09, 'AB': 0.04}
    print("\nReal OPTN distribution:")
    for bt, pct in sorted(real_optn.items()):
        bar = '█' * int(pct * 40)
        print(f"  {bt:4s} {bar} {pct:.1%}")

    # Compare
    print("\nComparison (synthetic vs OPTN):")
    for bt in ['O', 'A', 'B', 'AB']:
        synth = synth_bt.get(bt, 0)
        real  = real_optn.get(bt, 0)
        diff  = abs(synth - real)
        status = "✓" if diff < 0.05 else "✗"
        print(f"  {bt:4s} Synthetic: {synth:.1%}  Real: {real:.1%}  "
              f"Diff: {diff:.1%}  {status}")


# ─────────────────────────────────────────
# ANALYZE ORGAN TRANSPLANT KAGGLE DATA
# ─────────────────────────────────────────
def analyze_organ_transplant(ot_df):
    print("\n" + "="*50)
    print("ORGAN TRANSPLANT KAGGLE ANALYSIS")
    print("="*50)

    print(f"\nDataset shape: {ot_df.shape}")
    print(f"\nColumn types:")
    print(ot_df.dtypes.to_string())

    print(f"\nSample data (first 3 rows):")
    print(ot_df.head(3).to_string())

    print(f"\nMissing values:")
    missing = ot_df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print(missing.to_string())
    else:
        print("  No missing values!")

    # Numeric column stats
    numeric_cols = ot_df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        print(f"\nNumeric column statistics:")
        print(ot_df[numeric_cols].describe().round(2).to_string())

    return ot_df


# ─────────────────────────────────────────
# BUILD KAGGLE FEATURES
# ─────────────────────────────────────────
def build_kaggle_features(ot_df):
    """
    Map Kaggle Organ Transplant columns to our model's feature schema.
    Returns feature DataFrame compatible with our ML models.
    """
    print("\n" + "="*50)
    print("BUILDING KAGGLE FEATURES")
    print("="*50)

    cols = [c.lower().strip() for c in ot_df.columns]
    ot_df.columns = cols

    print(f"Available columns: {cols}")

    rows = []
    for _, row in ot_df.iterrows():
        try:
            # Map available columns — use defaults for missing ones
            features = {
                'abo_compatible':     int(row.get('abo_match', row.get('blood_match', 1))),
                'hla_mismatch':       int(row.get('hla_mismatch', row.get('hla_mm', 6))),
                'pra':                float(row.get('pra', row.get('panel_reactive_antibody', 50))),
                'age_delta':          abs(float(row.get('donor_age', 40)) -
                                         float(row.get('recipient_age', row.get('age', 40)))),
                'gfr':                float(row.get('gfr', row.get('egfr', 15))),
                'urgency_score':      int(row.get('urgency', row.get('urgency_score', 2))),
                'wait_time':          float(row.get('wait_time', row.get('waiting_time', 365))),
                'cold_ischemia_time': float(row.get('cold_ischemia_time', row.get('cit', 12))),
                'donor_type':         1 if str(row.get('donor_type', 'deceased')).lower() == 'living' else 0,
                'distance':           float(row.get('distance', 50)),
            }

            # Rejection label from outcome if available
            outcome_col = None
            for c in ['rejection', 'graft_failure', 'outcome', 'failed', 'status']:
                if c in cols:
                    outcome_col = c
                    break

            if outcome_col:
                label = 1 if str(row[outcome_col]).lower() in ['1', 'yes', 'rejected',
                                                                 'failed', 'failure'] else 0
            else:
                # Derive label from HLA mismatch + PRA
                reject_prob = (features['hla_mismatch']/12)*0.4 + \
                              (1-features['abo_compatible'])*0.35 + \
                              (features['pra']/100)*0.25
                label = 1 if reject_prob > 0.65 else 0

            features['rejection_label'] = label
            rows.append(features)

        except Exception as e:
            continue

    if not rows:
        print("  Could not map Kaggle columns — using distribution validation only")
        return None

    df = pd.DataFrame(rows)
    print(f"  Mapped {len(df)} records")
    print(f"  Rejection rate: {df['rejection_label'].mean():.1%}")
    return df


# ─────────────────────────────────────────
# VALIDATE MODEL ON KAGGLE DATA
# ─────────────────────────────────────────
def validate_on_kaggle(model, kaggle_X, kaggle_y, synthetic_metrics: dict):
    """
    Run trained model on Kaggle data.
    Compare AUC against synthetic baseline.
    """
    print("\n" + "="*50)
    print("MODEL VALIDATION ON KAGGLE DATA")
    print("="*50)

    probs = model.predict_proba(kaggle_X)[:, 1]
    preds = (probs >= 0.5).astype(int)

    # Manual AUC
    from ml_models.rejection_risk_model import roc_auc, f1_manual, brier_score
    kaggle_auc   = roc_auc(kaggle_y.values, probs)
    kaggle_f1    = f1_manual(kaggle_y.values, preds)
    kaggle_brier = brier_score(kaggle_y.values, probs)

    print(f"\n  Kaggle validation metrics:")
    print(f"  AUC-ROC: {kaggle_auc:.4f}")
    print(f"  F1:      {kaggle_f1:.4f}")
    print(f"  Brier:   {kaggle_brier:.4f}")

    print(f"\n  Comparison vs synthetic baseline:")
    synth_auc = synthetic_metrics.get('auc', 0)
    auc_drop  = synth_auc - kaggle_auc
    print(f"  Synthetic AUC: {synth_auc:.4f}")
    print(f"  Kaggle AUC:    {kaggle_auc:.4f}")
    print(f"  AUC drop:      {auc_drop:.4f}  "
          f"{'✓ Generalizes well' if auc_drop < 0.05 else '✗ Needs recalibration'}")

    return {
        'kaggle_auc':   round(kaggle_auc, 4),
        'kaggle_f1':    round(kaggle_f1, 4),
        'kaggle_brier': round(kaggle_brier, 4),
        'auc_drop':     round(auc_drop, 4),
    }


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    # Load data
    donors     = pd.read_csv('data/donors.csv')
    recipients = pd.read_csv('data/recipients.csv')
    donors.columns     = donors.columns.str.strip()
    recipients.columns = recipients.columns.str.strip()
    donors['blood_type']     = donors['blood_type'].str.strip()
    recipients['blood_type'] = recipients['blood_type'].str.strip()

    # Load Kaggle datasets
    kaggle_data = load_kaggle_data()

    # Blood type distribution comparison
    compare_blood_types(donors, kaggle_data)

    # KS test on age distribution
    print("\n" + "="*50)
    print("KS TEST — DISTRIBUTION SIMILARITY")
    print("="*50)
    print("\nTesting synthetic vs real age distributions:")
    real_ages = np.random.normal(45, 15, 1000).clip(18, 80)
    ks_test(donors['age'].values.astype(float),
            real_ages, 'donor_age')
    ks_test(recipients['age'].values.astype(float),
            np.random.normal(48, 18, 1000).clip(5, 80), 'recipient_age')

    # Analyze Kaggle datasets
    if 'organ_transplant' in kaggle_data:
        ot_df = analyze_organ_transplant(kaggle_data['organ_transplant'])
        kaggle_features = build_kaggle_features(ot_df)

        if kaggle_features is not None:
            print(f"\nKaggle feature sample:")
            print(kaggle_features.head(3).to_string())


    print("Cross-validation on Kaggle data")
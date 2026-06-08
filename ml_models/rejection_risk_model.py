import numpy as np
import pandas as pd
import json
import sys
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,
                             recall_score, brier_score_loss,
                             confusion_matrix, classification_report)
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import shap
import optuna
import warnings
warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ─────────────────────────────────────────
# FEATURE ENGINEERING
# ─────────────────────────────────────────
def compute_hla_mismatches(donor_hla: dict, recipient_hla: dict) -> int:
    mismatches = 0
    for locus in ['A', 'B', 'C', 'DR', 'DQ', 'DP']:
        d = set(donor_hla.get(locus, []))
        r = set(recipient_hla.get(locus, []))
        mismatches += len(d.symmetric_difference(r))
    return mismatches

def abo_compatible(donor_blood: str, recipient_blood: str) -> int:
    ABO = {
        'O':  ['O', 'A', 'B', 'AB'],
        'A':  ['A', 'AB'],
        'B':  ['B', 'AB'],
        'AB': ['AB'],
    }
    return 1 if recipient_blood in ABO.get(donor_blood, []) else 0

def build_features(donors: pd.DataFrame,
                   recipients: pd.DataFrame) -> pd.DataFrame:
    print("Building features...")
    rows = []
    for _, d in donors.iterrows():
        for _, r in recipients.iterrows():
            try:
                d_hla = json.loads(d['hla']) if isinstance(d['hla'], str) else d['hla']
                r_hla = json.loads(r['hla']) if isinstance(r['hla'], str) else r['hla']
            except:
                continue

            abo = abo_compatible(str(d['blood_type']).strip(),
                                 str(r['blood_type']).strip())

            hla_mm = compute_hla_mismatches(d_hla, r_hla)
            age_delta = abs(int(d['age']) - int(r['age']))
            dist = np.sqrt((float(d['location_lat']) - float(r['location_lat']))**2 +
                           (float(d['location_lon']) - float(r['location_lon']))**2)

            # Rejection label:
            # High HLA mismatch + ABO incompatible + high PRA = rejection
            pra = float(r['pra'])
            reject_prob = (hla_mm / 12) * 0.4 + (1 - abo) * 0.35 + (pra / 100) * 0.25
            label = 1 if reject_prob > 0.5 else 0

            rows.append({
                'abo_compatible':     abo,
                'hla_mismatch':       hla_mm,
                'pra':                pra,
                'age_delta':          age_delta,
                'gfr':                int(r['gfr']),
                'urgency_score':      int(r['urgency_score']),
                'wait_time':          int(r['wait_time']),
                'cold_ischemia_time': int(d['cold_ischemia_time']),
                'donor_type':         1 if str(d['donor_type']).strip() == 'living' else 0,
                'distance':           round(dist, 4),
                'rejection_label':    label,
            })

    return pd.DataFrame(rows)


# ─────────────────────────────────────────
# TRAIN ALL MODELS
# ─────────────────────────────────────────
def train_models(X: pd.DataFrame, y: pd.Series):
    print("\n" + "="*50)
    print("TRAINING ML MODELS")
    print("="*50)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # ── 1. Logistic Regression ──
    print("\n[1] Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_auc = cross_val_score(lr, X_scaled, y, cv=skf,
                             scoring='roc_auc').mean()
    lr_f1  = cross_val_score(lr, X_scaled, y, cv=skf,
                             scoring='f1').mean()
    lr.fit(X_scaled, y)
    lr_probs = lr.predict_proba(X_scaled)[:, 1]
    lr_brier = brier_score_loss(y, lr_probs)
    print(f"   AUC-ROC: {lr_auc:.4f} | F1: {lr_f1:.4f} | Brier: {lr_brier:.4f}")
    results['logistic_regression'] = {
        'model': lr, 'scaler': scaler,
        'auc': round(lr_auc, 4), 'f1': round(lr_f1, 4),
        'brier': round(lr_brier, 4)
    }

    # ── 2. Random Forest ──
    print("\n[2] Random Forest...")
    rf = RandomForestClassifier(n_estimators=200, random_state=42,
                                class_weight='balanced', n_jobs=-1)
    rf_auc = cross_val_score(rf, X, y, cv=skf,
                             scoring='roc_auc').mean()
    rf_f1  = cross_val_score(rf, X, y, cv=skf,
                             scoring='f1').mean()
    rf.fit(X, y)
    rf_probs = rf.predict_proba(X)[:, 1]
    rf_brier = brier_score_loss(y, rf_probs)
    print(f"   AUC-ROC: {rf_auc:.4f} | F1: {rf_f1:.4f} | Brier: {rf_brier:.4f}")
    results['random_forest'] = {
        'model': rf,
        'auc': round(rf_auc, 4), 'f1': round(rf_f1, 4),
        'brier': round(rf_brier, 4)
    }

    # ── 3. XGBoost with Optuna tuning ──
    print("\n[3] XGBoost + Optuna tuning...")

    def objective(trial):
        params = {
            'n_estimators':      trial.suggest_int('n_estimators', 100, 500),
            'max_depth':         trial.suggest_int('max_depth', 3, 8),
            'learning_rate':     trial.suggest_float('learning_rate', 0.01, 0.3),
            'subsample':         trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree':  trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'scale_pos_weight':  trial.suggest_float('scale_pos_weight', 1, 5),
            'random_state': 42,
            'eval_metric': 'logloss',
        }
        model = xgb.XGBClassifier(**params)
        score = cross_val_score(model, X, y, cv=3,
                                scoring='roc_auc').mean()
        return score

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=20)

    best_params = study.best_params
    best_params['random_state'] = 42
    xgb_model = xgb.XGBClassifier(**best_params)
    xgb_auc = cross_val_score(xgb_model, X, y, cv=skf,
                              scoring='roc_auc').mean()
    xgb_f1  = cross_val_score(xgb_model, X, y, cv=skf,
                              scoring='f1').mean()
    xgb_model.fit(X, y)
    xgb_probs = xgb_model.predict_proba(X)[:, 1]
    xgb_brier = brier_score_loss(y, xgb_probs)
    print(f"   Best params: {best_params}")
    print(f"   AUC-ROC: {xgb_auc:.4f} | F1: {xgb_f1:.4f} | Brier: {xgb_brier:.4f}")
    results['xgboost'] = {
        'model': xgb_model, 'best_params': best_params,
        'auc': round(xgb_auc, 4), 'f1': round(xgb_f1, 4),
        'brier': round(xgb_brier, 4)
    }

    return results, X, y, scaler


# ─────────────────────────────────────────
# SHAP EXPLAINABILITY
# ─────────────────────────────────────────
def explain_with_shap(model, X: pd.DataFrame):
    print("\n[SHAP] Computing feature importance...")
    explainer   = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Mean absolute SHAP values = global importance
    importance = pd.DataFrame({
        'feature':    X.columns,
        'importance': np.abs(shap_values).mean(axis=0)
    }).sort_values('importance', ascending=False)

    print("\nTop feature importances (SHAP):")
    for _, row in importance.iterrows():
        bar = '█' * int(row['importance'] * 100)
        print(f"  {row['feature']:25s} {bar} {row['importance']:.4f}")

    return shap_values, importance


# ─────────────────────────────────────────
# CALIBRATION CHECK
# ─────────────────────────────────────────
def check_calibration(model, X, y, model_name: str):
    print(f"\n[Calibration] {model_name}...")
    probs = model.predict_proba(X)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y, probs, n_bins=10)

    print(f"  Predicted vs Actual probability (10 bins):")
    for pred, actual in zip(mean_pred, fraction_pos):
        diff = abs(pred - actual)
        status = "✓" if diff < 0.1 else "✗"
        print(f"  Predicted: {pred:.2f} | Actual: {actual:.2f} | "
              f"Diff: {diff:.2f} {status}")

    calibrated = CalibratedClassifierCV(model, cv='prefit', method='isotonic')
    calibrated.fit(X, y)
    cal_probs = calibrated.predict_proba(X)[:, 1]
    cal_brier = brier_score_loss(y, cal_probs)
    print(f"  Calibrated Brier score: {cal_brier:.4f}")
    return calibrated


# ─────────────────────────────────────────
# SURVIVAL ANALYSIS
# ─────────────────────────────────────────
def survival_analysis(recipients: pd.DataFrame):
    print("\n" + "="*50)
    print("SURVIVAL ANALYSIS")
    print("="*50)

    try:
        from lifelines import CoxPHFitter, KaplanMeierFitter

        # Simulate survival data from recipient features
        np.random.seed(42)
        n = len(recipients)

        # Survival time in days (1yr = 365, 5yr = 1825)
        base_survival = 365 + (recipients['gfr'].values * 20)
        noise = np.random.normal(0, 100, n)
        survival_time = np.clip(base_survival + noise, 30, 1825)

        # Event: 1 = graft failure, 0 = censored (still alive)
        event = (survival_time < 800).astype(int)

        survival_df = pd.DataFrame({
            'duration':      survival_time,
            'event':         event,
            'gfr':           recipients['gfr'].values,
            'pra':           recipients['pra'].values,
            'urgency_score': recipients['urgency_score'].values,
            'wait_time':     recipients['wait_time'].values,
            'age':           recipients['age'].values,
        })

        # Kaplan-Meier
        print("\n[Kaplan-Meier] Overall survival estimate...")
        kmf = KaplanMeierFitter()
        kmf.fit(survival_df['duration'], event_observed=survival_df['event'])
        prob_1yr = kmf.predict(365)
        prob_5yr = kmf.predict(1825)
        print(f"  1-year graft survival probability:  {prob_1yr:.1%}")
        print(f"  5-year graft survival probability:  {prob_5yr:.1%}")

        # Cox Proportional Hazards
        print("\n[Cox PH] Fitting survival regression...")
        cph = CoxPHFitter()
        cph.fit(survival_df, duration_col='duration', event_col='event')
        print("\n  Cox model summary (top risk factors):")
        summary = cph.summary[['coef', 'exp(coef)', 'p']].head(5)
        print(summary.to_string())

        print(f"\n  Concordance index: {cph.concordance_index_:.4f}")
        print("  (>0.6 = good, >0.7 = strong predictive power)")

        return kmf, cph, survival_df

    except ImportError:
        print("lifelines not installed. Run: pip install lifelines")
        return None, None, None


# ─────────────────────────────────────────
# FINAL SUMMARY
# ─────────────────────────────────────────
def print_summary(results: dict):
    print("\n" + "="*50)
    print("FINAL MODEL COMPARISON")
    print("="*50)
    print(f"\n{'Model':25s} {'AUC-ROC':>10} {'F1':>10} {'Brier':>10}")
    print("-"*55)
    for name, r in results.items():
        if 'auc' in r:
            print(f"{name:25s} {r['auc']:>10.4f} {r['f1']:>10.4f} {r['brier']:>10.4f}")
    print("-"*55)
    print("Lower Brier = better calibration")
    print("Higher AUC  = better discrimination")


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────
if __name__ == '__main__':
    donors     = pd.read_csv('data/donors.csv')
    recipients = pd.read_csv('data/recipients.csv')
    donors.columns     = donors.columns.str.strip()
    recipients.columns = recipients.columns.str.strip()
    donors['blood_type']     = donors['blood_type'].str.strip()
    recipients['blood_type'] = recipients['blood_type'].str.strip()

    # Use sample for speed
    d_sample = donors.head(50)
    r_sample = recipients.head(100)

    print(f"Building features for {len(d_sample)} donors x {len(r_sample)} recipients...")
    df = build_features(d_sample, r_sample)
    print(f"Dataset size: {len(df)} pairs")
    print(f"Rejection rate: {df['rejection_label'].mean():.1%}")

    X = df.drop('rejection_label', axis=1)
    y = df['rejection_label']

    # Train all models
    results, X, y, scaler = train_models(X, y)

    # SHAP on XGBoost
    shap_values, importance = explain_with_shap(
        results['xgboost']['model'], X
    )

    # Calibration check
    check_calibration(results['xgboost']['model'], X, y, 'XGBoost')

    # Survival analysis
    kmf, cph, survival_df = survival_analysis(r_sample)

    # Final summary
    print_summary(results)

  
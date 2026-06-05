# Organ Match AI — Project Notes

## Data Strategy

- Training data: synthetic (500 donors, 1000 recipients)
- Generated using real OPTN/UNOS blood type distributions
- Validation data: real Kaggle datasets (see below)

## Kaggle Datasets (download and save to data/kaggle/)

1. NHS Organ Donation Dataset
   - URL: https://www.kaggle.com/datasets/patricklford/nhs-organ-donation
   - Save as: data/kaggle/nhs_organ_donation.csv

2. Organ Transplant Dataset
   - URL: https://www.kaggle.com/datasets/fkshaikh/organ-transplant-dataset
   - Save as: data/kaggle/organ_transplant.csv

## Validation Plan

- Day 1: train all models on synthetic data, save baseline metrics
- Day 2 Phase 5: load Kaggle data, run KS test vs synthetic distributions
- Day 2 Phase 6: re-run models on Kaggle data, compare AUC/F1/Brier

## Target Metrics

- XGBoost AUC-ROC > 0.82
- F1 Score > 0.78
- Kaggle validation AUC drop < 0.05
- Stable matching: 100% stability, >85% utilization

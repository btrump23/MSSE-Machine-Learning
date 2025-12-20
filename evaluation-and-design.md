# Evaluation and Design

## 1. Problem Overview

The objective of this project is to build a machine learning system capable of classifying software samples as **malware (1)** or **goodware (0)** based on extracted static features. The task is framed as a binary classification problem with an emphasis on high recall and F1-score, as false negatives (malware misclassified as benign) are particularly costly in real-world security settings.

---

## 2. Dataset and Experimental Setup

The dataset consists of engineered static features extracted from executable files. Features include numerical indicators derived from file structure and metadata. A stratified split was used to preserve class balance.

### Data Splitting Strategy
- **80% training set**
- **20% hold-out test set (locked and never reused)**

The hold-out test set was reserved strictly for final evaluation after all model selection and hyperparameter decisions were completed.

---

## 3. Exploratory Data Analysis (EDA)

Exploratory analysis was conducted on the training data only and included:

- Verification of class balance
- Inspection of missing values
- Review of feature distributions

Several columns were found to contain **100% missing values** (e.g. `FormatedTimeDateStamp`, `ImportedDlls`, `ImportedSymbols`, `SHA1`) and were removed prior to modeling. After cleaning, **22–23 numeric features** were consistently used across experiments.

---

## 4. Preprocessing and Data Leakage Prevention

To ensure robust evaluation and prevent data leakage:

- All preprocessing steps (median imputation and standard scaling) were implemented using an **sklearn Pipeline**
- Preprocessing was **fit only on training folds** during cross-validation
- No statistics from validation or test data were used during training

This pipeline-based approach guarantees that transformations are learned exclusively from training data, preventing information leakage.

---

## 5. Model Selection and Cross-Validation

A range of models were evaluated using **stratified 10-fold cross-validation** on the training set only. Performance was measured using accuracy, precision, recall, and F1-score.

### Models Evaluated
- Logistic Regression (baseline)
- Decision Tree
- Random Forest
- XGBoost
- LightGBM
- CatBoost
- PyTorch Multi-Layer Perceptron (MLP)

### Cross-Validation Summary (Training Only)

| Model | Accuracy (CV) | F1-score (CV) | Notes |
|-----|-------------|--------------|------|
| Logistic Regression | ~0.80 | ~0.82 | Linear baseline |
| Decision Tree | ~0.98 | ~0.98 | High variance |
| Random Forest | ~0.99 | ~0.99 | Strong but heavier |
| XGBoost | ~0.99 | ~0.99 | Excellent performance |
| **LightGBM** | **~0.99** | **~0.99** | Best balance of speed & accuracy |
| CatBoost | ~0.99 | ~0.99 | Required manual CV |
| PyTorch MLP | ~0.94 | ~0.95 | Lower performance |

---

## 6. Final Model Selection

**LightGBM** was selected as the final model due to:

- Consistently high cross-validation performance
- Low variance across folds
- Fast training and inference
- Seamless integration with sklearn Pipelines
- Strong performance on unseen data

---

## 7. Hold-out Test Evaluation

After model selection, LightGBM was evaluated **once** on the locked hold-out test set.

### Final Hold-out Results (LightGBM)

- **Accuracy:** 0.9879
- **Precision:** 0.9894
- **Recall:** 0.9899
- **F1-score:** 0.9896

#### Confusion Matrix (rows = true, cols = predicted)

[[4161 63]
[ 60 5863]]

These results confirm that the model generalizes well and maintains strong recall and precision on unseen data.

---

## 8. Model Packaging and Reusability

The final LightGBM model was packaged as a reusable sklearn Pipeline and saved using `joblib`:

artifacts/lgbm_pipeline.joblib


The artifact contains:
- The full preprocessing pipeline
- The trained LightGBM classifier
- The list of training feature names

---

## 9. Prediction Wrapper

A standalone prediction wrapper (`predict.py`) was implemented to:

- Load the packaged pipeline artifact
- Align incoming CSV inputs to the exact training feature schema
- Generate predictions and malware probabilities
- Optionally evaluate performance when labels are present

The wrapper was validated by running predictions on the hold-out test CSV and exporting results to `artifacts/test_predictions.csv`, reproducing the expected hold-out metrics.

---

## 10. Summary

This project demonstrates an end-to-end machine learning workflow including careful data handling, leakage-free evaluation, robust model comparison, and production-ready packaging. The final LightGBM model achieves strong performance while remaining efficient and deployable, making it well-suited for real-world malware detection scenarios.


The artifact contains:
- The full preprocessing pipeline
- The trained LightGBM classifier
- The list of training feature names

---

## 9. Prediction Wrapper

A standalone prediction wrapper (`predict.py`) was implemented to:

- Load the packaged pipeline artifact
- Align incoming CSV inputs to the exact training feature schema
- Generate predictions and malware probabilities
- Optionally evaluate performance when labels are present

The wrapper was validated by running predictions on the hold-out test CSV and exporting results to `artifacts/test_predictions.csv`, reproducing the expected hold-out metrics.

---

## 10. Summary

This project demonstrates an end-to-end machine learning workflow including careful data handling, leakage-free evaluation, robust model comparison, and production-ready packaging. The final LightGBM model achieves strong performance while remaining efficient and deployable, making it well-suited for real-world malware detection scenarios.



# Diabetes Risk Prediction

An end-to-end Python workflow for three supervised-learning tasks:

1. Binary classification of diagnosed diabetes.
2. Multiclass classification of diabetes stage.
3. Regression of a continuous diabetes risk score.

This repository refactors the original notebook into a reproducible analysis script. Preprocessing and PCA are contained inside each modelling pipeline so they are fitted within cross-validation folds rather than before validation.

## Methods

- Standardise category labels and remove invalid records.
- Impute numerical values with medians and categorical values with modes.
- Explore distributions, correlations, and diagnosis-related patterns.
- Compare Logistic Regression, Random Forest, SVM, and Linear Regression as appropriate to each task.
- Use one-hot encoding, scaling, PCA, stratified or shuffled five-fold cross-validation, held-out evaluation, and confusion matrices.

## Data

The dataset contains 5,000 synthetically generated patient-style records designed from statistical distributions inspired by medical research. They are not real patient records. See `docs/project-scope.md` for the project scope and `data/README.md` for dataset context.

The analysis expects `data/Diabetes.csv` and `data/Diabetes_test.csv`.

## Reproduce

```bash
python -m pip install -r requirements.txt
python src/diabetes_analysis.py
```

Figures are displayed locally. No patient-level output is saved by the script.

## Original reported results

The original analysis reported approximately 62% accuracy and 84% recall for binary classification, weaker multiclass performance under severe class imbalance, and Linear Regression R² of 0.98 for risk-score prediction. These are historical results, not newly verified benchmarks. The high regression score may reflect a deterministic relationship between the target and inputs, so the analysis should not be interpreted as clinical validation.

## Responsible-use note

This is a synthetic-data demonstration, not a medical device or diagnostic tool. Model outputs must not be used for diagnosis, treatment, or patient triage.

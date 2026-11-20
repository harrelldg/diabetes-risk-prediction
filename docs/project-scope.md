# Project Scope

## Dataset context

The dataset contains approximately 5,000 privacy-preserving, synthetically generated patient-style records for diabetes risk analysis. Features include demographics, lifestyle habits, family history, and clinical measurements. Data cleaning and preprocessing are required before modelling.

Key fields include `patient_id`, age, gender, ethnicity, education, income, employment, smoking, alcohol consumption, physical activity, diet, sleep, screen time, family history, BMI, waist-to-hip ratio, blood pressure, heart rate, cholesterol, insulin, `diabetes_risk_score`, `diabetes_stage`, and `diagnosed_diabetes`.

## Workflow

### 1. Data exploration

Inspect distributions and data types; identify missing values, duplicates, outliers, and malformed categories; document all cleaning decisions and record counts. An IQR rule using 2.5 times the interquartile range is used for extreme-outlier review.

### 2. Data visualisation

Use tables and figures to examine variable distributions, relationships, and meaningful patterns.

### 3. Modelling

Build and compare at least three models using five-fold cross-validation:

- Binary classification for `diagnosed_diabetes`.
- Multiclass classification for `diabetes_stage`.
- Regression for `diabetes_risk_score`.

Evaluate on `Diabetes_test.csv`, apply dimensionality reduction, visualise results, and use appropriate metrics including accuracy and confusion matrices.

### 4. Summary

Summarise findings and draw conclusions.

## Source

Accessible text summary of the original project instructions. Submission-specific identifiers and administrative details are omitted from this public summary.

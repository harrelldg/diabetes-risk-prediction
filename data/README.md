# Dataset

`Diabetes.csv` contains approximately 5,000 synthetically generated patient-style records used for model training and cross-validation.

`Diabetes_test.csv` is the held-out evaluation file expected by `src/diabetes_analysis.py`.

The records are synthetic and privacy-preserving. They are not real patient data and must not be treated as clinically validated observations.

## Key fields

- Targets: `diagnosed_diabetes`, `diabetes_stage`, and `diabetes_risk_score`.
- Demographics: age, gender, ethnicity, education, income, and employment.
- Lifestyle: smoking, alcohol consumption, physical activity, diet, sleep, and screen time.
- Medical indicators: family history, BMI, waist-to-hip ratio, blood pressure, heart rate, cholesterol, and insulin.

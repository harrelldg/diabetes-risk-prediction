"""Diabetes risk modelling: cleaned and reproducible academic analysis."""

from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore", category=FutureWarning)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
TARGETS = ["diagnosed_diabetes", "diabetes_stage", "diabetes_risk_score"]


def load_data():
    train = pd.read_csv(DATA_DIR / "Diabetes.csv")
    test = pd.read_csv(DATA_DIR / "Diabetes_test.csv")
    for frame in (train, test):
        frame.columns = frame.columns.str.strip().str.lower().str.replace(" ", "_", regex=False)
    return train, test


def clean_training_data(frame):
    frame = frame.copy().drop_duplicates().reset_index(drop=True)
    frame["education_level"] = frame["education_level"].replace({"University": "Graduate"})
    frame["smoking_status"] = frame["smoking_status"].replace({"Mever": "Never"})
    frame["diabetes_stage"] = frame["diabetes_stage"].replace({"Type 22": "Type 2"})
    frame = frame[~frame["diabetes_stage"].isin(["Type3"])]
    frame = frame[frame["family_history_diabetes"].isin([0, 1])]
    frame = frame[frame["diagnosed_diabetes"].isin([0, 1])]
    return frame.reset_index(drop=True)


def split_features(frame, target):
    excluded = ["patient_id", *TARGETS]
    return frame.drop(columns=[c for c in excluded if c in frame.columns]), frame[target]


def make_preprocessor(X):
    categorical = X.select_dtypes(include=["object", "category"]).columns
    numerical = X.select_dtypes(include=np.number).columns
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numerical),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])


def make_pipeline(X, model):
    return Pipeline([
        ("preprocess", make_preprocessor(X)),
        ("pca", PCA(n_components=0.95, svd_solver="full")),
        ("model", model),
    ])


def explore_data(train):
    numeric = train.select_dtypes(include=np.number).columns
    numeric = [c for c in numeric if c not in ["patient_id", *TARGETS]]
    categorical = [
        "gender", "ethnicity", "education_level", "income_level",
        "employment_status", "smoking_status", "diabetes_stage"
    ]

    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
    for ax, column in zip(axes.flat, categorical):
        sns.countplot(data=train, x=column, hue="diagnosed_diabetes", ax=ax)
        ax.tick_params(axis="x", rotation=45)
        ax.set_title(f"{column} by diagnosis")
    for ax in list(axes.flat)[len(categorical):]:
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    if numeric:
        fig, axes = plt.subplots(4, 4, figsize=(20, 16))
        for ax, column in zip(axes.flat, numeric):
            sns.boxplot(data=train, x="diagnosed_diabetes", y=column, ax=ax, color="skyblue")
            ax.set_title(f"{column} by diagnosis")
        for ax in list(axes.flat)[len(numeric):]:
            ax.axis("off")
        plt.tight_layout()
        plt.show()

    corr_columns = numeric + ["diagnosed_diabetes", "family_history_diabetes"]
    plt.figure(figsize=(12, 8))
    sns.heatmap(train[corr_columns].corr(), cmap="coolwarm", center=0)
    plt.title("Feature Correlation Matrix")
    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred, title):
    labels = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    plt.figure(figsize=(6, 5))
    sns.heatmap(confusion_matrix(y_true, y_pred, labels=labels), annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def evaluate_classification(train, test, target, models):
    X_train, y_train = split_features(train, target)
    X_test, y_test = split_features(test, target)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    rows = []

    for name, model in models.items():
        pipeline = make_pipeline(X_train, model)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        rows.append({
            "model": name,
            "cv_accuracy": scores["test_accuracy"].mean(),
            "test_accuracy": accuracy_score(y_test, predicted),
            "test_precision": precision_score(y_test, predicted, average="weighted", zero_division=0),
            "test_recall": recall_score(y_test, predicted, average="weighted", zero_division=0),
            "test_f1": f1_score(y_test, predicted, average="weighted", zero_division=0),
        })
        print(f"\n{name}: {target}")
        print(classification_report(y_test, predicted, zero_division=0))
        plot_confusion(y_test, predicted, f"{name}: {target}")
    return pd.DataFrame(rows)


def evaluate_regression(train, test):
    X_train, y_train = split_features(train, "diabetes_risk_score")
    X_test, y_test = split_features(test, "diabetes_risk_score")
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        pipeline = make_pipeline(X_train, model)
        scores = cross_validate(pipeline, X_train, y_train, cv=cv, scoring=["r2", "neg_mean_squared_error"], n_jobs=-1)
        pipeline.fit(X_train, y_train)
        predicted = pipeline.predict(X_test)
        rows.append({
            "model": name,
            "cv_r2": scores["test_r2"].mean(),
            "test_r2": r2_score(y_test, predicted),
            "test_rmse": np.sqrt(mean_squared_error(y_test, predicted)),
        })
        plt.figure(figsize=(6, 5))
        plt.scatter(y_test, predicted, alpha=0.6)
        low = min(y_test.min(), predicted.min())
        high = max(y_test.max(), predicted.max())
        plt.plot([low, high], [low, high], "k--")
        plt.xlabel("Actual risk score")
        plt.ylabel("Predicted risk score")
        plt.title(name)
        plt.tight_layout()
        plt.show()
    return pd.DataFrame(rows)


def main():
    train, test = load_data()
    train = clean_training_data(train)
    print("Training shape after cleaning:", train.shape)
    explore_data(train)

    binary = evaluate_classification(train, test, "diagnosed_diabetes", {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    })
    multiclass = evaluate_classification(train, test, "diabetes_stage", {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM": SVC(kernel="rbf", C=1.0, random_state=42),
    })
    regression = evaluate_regression(train, test)
    print("\nBinary classification\n", binary.round(3).to_string(index=False))
    print("\nMulticlass classification\n", multiclass.round(3).to_string(index=False))
    print("\nRisk-score regression\n", regression.round(3).to_string(index=False))


if __name__ == "__main__":
    main()

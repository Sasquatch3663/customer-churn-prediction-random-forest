import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay,
    PrecisionRecallDisplay, classification_report
)
from sklearn.ensemble import RandomForestClassifier
import joblib

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")


# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "telco_churn.csv")
FIG_DIR = os.path.join(BASE_DIR, "reports", "figures")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------
# Helper plot saving
# -------------------------
def save_fig(name: str):
    path = os.path.join(FIG_DIR, name)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()
    print(f"[SAVED] {path}")


# -------------------------
# Load dataset
# -------------------------
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}\n\n"
        "Download Telco Customer Churn dataset and place it as:\n"
        "data/telco_churn.csv"
    )

df = pd.read_csv(DATA_PATH)

# -------------------------
# Basic cleaning
# -------------------------
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

if "customerID" in df.columns:
    df = df.drop(columns=["customerID"])

# -------------------------
# Train/Test split
# -------------------------
X = df.drop(columns=["Churn"])
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

num_cols = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_cols = X_train.select_dtypes(include=["object", "bool"]).columns.tolist()

# -------------------------
# Preprocessing
# -------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, cat_cols),
    ],
    remainder="drop"
)

# ============================================================
# MODEL 1: Random Forest with class_weight balanced
# ============================================================
rf_balanced = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

pipe_balanced = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", rf_balanced)
])

pipe_balanced.fit(X_train, y_train)

y_pred = pipe_balanced.predict(X_test)
y_proba = pipe_balanced.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print("\n==============================")
print("MODEL 1: Random Forest (class_weight='balanced')")
print("==============================")
print(f"Accuracy  : {acc:.4f}")
print(f"Precision : {prec:.4f}")
print(f"Recall    : {rec:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {auc:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(values_format="d")
plt.title("Confusion Matrix — RF (class_weight='balanced')")
save_fig("01_confusion_matrix_rf_balanced.png")

RocCurveDisplay.from_predictions(y_test, y_proba)
plt.title("ROC Curve — RF (class_weight='balanced')")
save_fig("02_roc_curve_rf_balanced.png")

PrecisionRecallDisplay.from_predictions(y_test, y_proba)
plt.title("Precision-Recall Curve — RF (class_weight='balanced')")
save_fig("03_pr_curve_rf_balanced.png")


# ============================================================
# BONUS MODEL 2: Random Forest + SMOTE
# ============================================================
rf_smote = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

pipe_smote = ImbPipeline(steps=[
    ("preprocess", preprocessor),
    ("smote", SMOTE(random_state=42)),
    ("model", rf_smote)
])

pipe_smote.fit(X_train, y_train)

y_pred_s = pipe_smote.predict(X_test)
y_proba_s = pipe_smote.predict_proba(X_test)[:, 1]

acc_s = accuracy_score(y_test, y_pred_s)
prec_s = precision_score(y_test, y_pred_s)
rec_s = recall_score(y_test, y_pred_s)
f1_s = f1_score(y_test, y_pred_s)
auc_s = roc_auc_score(y_test, y_proba_s)

print("\n==============================")
print("MODEL 2: Random Forest + SMOTE")
print("==============================")
print(f"Accuracy  : {acc_s:.4f}")
print(f"Precision : {prec_s:.4f}")
print(f"Recall    : {rec_s:.4f}")
print(f"F1-score  : {f1_s:.4f}")
print(f"ROC-AUC   : {auc_s:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_s))

cm2 = confusion_matrix(y_test, y_pred_s)
disp2 = ConfusionMatrixDisplay(confusion_matrix=cm2)
disp2.plot(values_format="d")
plt.title("Confusion Matrix — RF + SMOTE")
save_fig("04_confusion_matrix_rf_smote.png")

RocCurveDisplay.from_predictions(y_test, y_proba_s)
plt.title("ROC Curve — RF + SMOTE")
save_fig("05_roc_curve_rf_smote.png")

PrecisionRecallDisplay.from_predictions(y_test, y_proba_s)
plt.title("Precision-Recall Curve — RF + SMOTE")
save_fig("06_pr_curve_rf_smote.png")


# ============================================================
# BONUS MODEL 3: Hyperparameter tuning (GridSearchCV)
# ============================================================
param_grid = {
    "model__n_estimators": [200, 400],
    "model__max_depth": [None, 8, 14],
    "model__min_samples_split": [2, 10],
    "model__min_samples_leaf": [1, 3],
}

rf_for_grid = RandomForestClassifier(
    random_state=42,
    n_jobs=-1,
    class_weight="balanced"
)

pipe_grid = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", rf_for_grid)
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid = GridSearchCV(
    estimator=pipe_grid,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

grid.fit(X_train, y_train)

print("\n==============================")
print("MODEL 3: Tuned Random Forest (GridSearchCV)")
print("==============================")
print("Best Params:", grid.best_params_)
print("Best CV ROC-AUC:", round(grid.best_score_, 4))

best_model = grid.best_estimator_

y_pred_g = best_model.predict(X_test)
y_proba_g = best_model.predict_proba(X_test)[:, 1]

acc_g = accuracy_score(y_test, y_pred_g)
prec_g = precision_score(y_test, y_pred_g)
rec_g = recall_score(y_test, y_pred_g)
f1_g = f1_score(y_test, y_pred_g)
auc_g = roc_auc_score(y_test, y_proba_g)

print("\nTest Metrics:")
print(f"Accuracy  : {acc_g:.4f}")
print(f"Precision : {prec_g:.4f}")
print(f"Recall    : {rec_g:.4f}")
print(f"F1-score  : {f1_g:.4f}")
print(f"ROC-AUC   : {auc_g:.4f}")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred_g))

cm3 = confusion_matrix(y_test, y_pred_g)
disp3 = ConfusionMatrixDisplay(confusion_matrix=cm3)
disp3.plot(values_format="d")
plt.title("Confusion Matrix — Tuned RF")
save_fig("07_confusion_matrix_rf_tuned.png")

RocCurveDisplay.from_predictions(y_test, y_proba_g)
plt.title("ROC Curve — Tuned RF")
save_fig("08_roc_curve_rf_tuned.png")

PrecisionRecallDisplay.from_predictions(y_test, y_proba_g)
plt.title("Precision-Recall Curve — Tuned RF")
save_fig("09_pr_curve_rf_tuned.png")


# ============================================================
# Feature importance (from tuned model)
# ============================================================
pre = best_model.named_steps["preprocess"]
ohe = pre.named_transformers_["cat"].named_steps["onehot"]
cat_feature_names = ohe.get_feature_names_out(cat_cols)

all_features = np.concatenate([np.array(num_cols), cat_feature_names])

rf_final = best_model.named_steps["model"]
importances = rf_final.feature_importances_

idx = np.argsort(importances)[::-1][:20]
plt.figure(figsize=(10, 6))
plt.barh(all_features[idx][::-1], importances[idx][::-1])
plt.title("Top 20 Feature Importances — Tuned Random Forest")
plt.xlabel("Importance")
save_fig("10_feature_importance_top20.png")


# ============================================================
# Save final model
# ============================================================
model_path = os.path.join(MODEL_DIR, "churn_rf_model.pkl")
joblib.dump(best_model, model_path)
print(f"\n[SAVED MODEL] {model_path}")

print("\nDone ✅ Check reports/figures/ for images.")

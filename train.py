import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# =========================================================
# LOAD DATASET
# =========================================================
df = pd.read_csv("student_performance.csv")

# =========================================================
# RENAME SHORT/UNCLEAR COLUMN NAMES
# =========================================================
rename_map = {
    "school_typ": "school_type",
    "parent_ed": "parent_education",
    "study_hou": "study_hours",
    "attendance": "attendance_percentage",
    "internet_a": "internet_access",
    "travel_tim": "travel_time",
    "extra_acti": "extra_activities",
    "study_met": "study_method",
    "math_scor": "math_score",
    "science_s": "science_score",
    "english_sc": "english_score",
    "overall_sc": "overall_score"
}

df = df.rename(columns=rename_map)

# =========================================================
# CLEAN DATA
# =========================================================
df = df.dropna()

# Remove ID column if present
if "student_id" in df.columns:
    df = df.drop(columns=["student_id"])

# =========================================================
# CREATE TARGET
# =========================================================
# We predict final_grade directly
target_col = "final_grade"

# Features should NOT include final_grade
X = df.drop(columns=["final_grade"])
y = df["final_grade"]

# =========================================================
# DROP LEAKAGE COLUMN IF PRESENT
# =========================================================
# overall_score is often too close to final_grade, so remove it
if "overall_score" in X.columns:
    X = X.drop(columns=["overall_score"])

# =========================================================
# CHECK COLUMN TYPES
# =========================================================
categorical_cols = [col for col in X.columns if X[col].dtype == "object"]

# =========================================================
# PREPROCESSOR
# =========================================================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ],
    remainder="passthrough"
)

# =========================================================
# MODEL PIPELINE
# =========================================================
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=200,
        random_state=42
    ))
])

# =========================================================
# TRAIN / TEST SPLIT
# =========================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# TRAIN MODEL
# =========================================================
model.fit(X_train, y_train)

# =========================================================
# EVALUATION
# =========================================================
y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# =========================================================
# SAVE MODEL
# =========================================================
pickle.dump(model, open("student_model.pkl", "wb"))

print("\nModel saved successfully as student_model.pkl")
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import os

# Load dataset (local repository copy or original path)
local_csv = os.path.join(os.path.dirname(os.path.abspath(__file__)), "CKD_10000_sample_synthetic.csv")
file_path = local_csv if os.path.exists(local_csv) else r"C:\Users\mnith\OneDrive\Attachments\ML PROJECT\CKD_10000_sample_synthetic.csv"
df = pd.read_csv(file_path)

print("CHRONIC KIDNEY DISEASE DATASET")
print(df.head())
print("\nDataset shape:", df.shape)

target = "ckd_present"

if target not in df.columns:
    raise ValueError(
        f"Target column '{target}' not found. "
        f"Available columns: {list(df.columns)}"
    )

# Remove rows with missing target values
df = df.dropna(subset=[target]).copy()

# Separate target and features
y = pd.to_numeric(df[target], errors="coerce")
valid = y.notna()

df = df.loc[valid].copy()
y = y.loc[valid].astype(int)

# Remove target, ID, and potential target-leakage column
drop_columns = [
    col for col in ["participant_id", "ckd_stage", target]
    if col in df.columns
]

X = df.drop(columns=drop_columns)

# Identify categorical and numerical columns
categorical_columns = X.select_dtypes(
    include=["object", "category", "bool"]
).columns.tolist()

numerical_columns = [
    col for col in X.columns
    if col not in categorical_columns
]

# Preprocess numerical features
numerical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="median")
    )
])

# Preprocess categorical features
categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(strategy="most_frequent")
    ),
    (
        "encoder",
        OneHotEncoder(handle_unknown="ignore")
    )
])

# Combine preprocessing steps
preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_columns),
    ("categorical", categorical_pipeline, categorical_columns)
])

# Create Random Forest model
model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            class_weight="balanced"
        )
    )
])

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate model
print("\nCKD PREDICTION RESULTS")
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

print(
    "\nAccuracy:",
    round(accuracy_score(y_test, predictions), 3)
)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        predictions,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, predictions))

print("\nClass labels: 0 = CKD absent, 1 = CKD present")



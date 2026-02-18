import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import os

# Create models folder if not exists
if not os.path.exists("models"):
    os.makedirs("models")

print("Loading dataset...")
df = pd.read_csv("data/creditcard.csv")

print("Dataset Loaded Successfully!")

# Features and Target
X = df.drop("Class", axis=1)
y = df["Class"]

# Handle Imbalanced Data using SMOTE
print("Applying SMOTE to balance dataset...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

# Train Model
print("Training Random Forest Model...")
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print("ROC AUC Score:", roc_auc_score(y_test, y_pred))

# Save Model
joblib.dump(model, "models/model.pkl")
print("\nModel saved as models/model.pkl")

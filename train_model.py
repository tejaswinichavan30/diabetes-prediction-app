"""
Trains a diabetes prediction model on the Pima Indians Diabetes dataset
and saves the fitted model + scaler as pickle files.

Run this once (or whenever you want to retrain):
    python train_model.py
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

# 1. Load data
df = pd.read_csv('diabetes.csv', header=None, names=COLUMNS)

# 2. Basic cleaning: several columns use 0 as a placeholder for "missing"
#    (a real Glucose/BMI/BloodPressure of 0 is medically impossible).
#    Replace those 0s with NaN, then impute with the column median.
zero_as_missing = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in zero_as_missing:
    df[col] = df[col].replace(0, np.nan)
    df[col] = df[col].fillna(df[col].median())

X = df.drop('Outcome', axis=1)
y = df['Outcome']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Scale features (important for consistent, well-calibrated predictions)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. Train model (RandomForest generally outperforms plain LogisticRegression
#    on this dataset and gives better probability estimates)
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=6,
    min_samples_leaf=3,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train_scaled, y_train)

# 6. Evaluate
y_pred = model.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)
cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5)

print(f"Test accuracy: {acc:.3f}")
print(f"5-fold CV accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")
print(classification_report(y_test, y_pred))

# 7. Save model + scaler + column order (so app.py always matches training)
with open('diabetes_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('columns.pkl', 'wb') as f:
    pickle.dump(list(X.columns), f)

print("\nSaved diabetes_model.pkl, scaler.pkl, columns.pkl")

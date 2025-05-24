import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load both CSVs
pushup_df = pd.read_csv("pushup_landmarks.csv")
non_pushup_df = pd.read_csv("non_pushup_landmarks.csv")

# Label them
pushup_df['label'] = 1
non_pushup_df['label'] = 0

# Combine datasets
df = pd.concat([pushup_df, non_pushup_df], ignore_index=True)

# Drop non-feature columns
X = df.drop(columns=["frame", "label"])
y = df['label']

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

# Train classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "pushup_detector.pkl")
joblib.dump(scaler, "scaler.pkl")

# Optional: Print accuracy
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

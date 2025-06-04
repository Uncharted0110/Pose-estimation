import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Get the directory where the script is located
script_dir = os.path.dirname(os.path.abspath(__file__))
planks_path = os.path.join(script_dir, "planks.csv")
non_planks_path = os.path.join(script_dir, "non_pushup_landmarks.csv")

# Load both CSVs
planks_df = pd.read_csv(planks_path)
non_planks_df = pd.read_csv(non_planks_path)

# Label them
planks_df['label'] = 1  # Planks
non_planks_df['label'] = 0  # Other actions

# Combine datasets
df = pd.concat([planks_df, non_planks_df], ignore_index=True)

# Drop non-feature columns
X = df.drop(columns=["frame", "label"])
y = df['label']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)

# Train the classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and scaler
joblib.dump(model, "planks_detector.pkl")
joblib.dump(scaler, "planks_scaler.pkl")

# Print performance
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

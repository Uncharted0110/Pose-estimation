import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load both CSVs
twists_df = pd.read_csv("all_plank/planks.csv")
non_twists_df = pd.read_csv("non_pushup_landmarks.csv")

# Label them
twists_df['label'] = 1  # Russian Twists
non_twists_df['label'] = 0  # Other actions

# Combine datasets
df = pd.concat([twists_df, non_twists_df], ignore_index=True)

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
joblib.dump(model, "twist_detector.pkl")
joblib.dump(scaler, "twist_scaler.pkl")

# Print performance
print("Train Accuracy:", model.score(X_train, y_train))
print("Test Accuracy:", model.score(X_test, y_test))

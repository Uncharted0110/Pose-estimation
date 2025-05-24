import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
df = pd.read_csv('pushup_dataset.csv')

# Drop the frame number and label (keep label separately if needed)
labels = df['label']
df = df.drop(columns=['frame', 'label'])

# Drop all visibility columns (every 4th column starting from 3)
v_cols = [col for i, col in enumerate(df.columns) if i % 4 == 3]
df = df.drop(columns=v_cols)

# Convert to numpy array (each row = flattened [x0, y0, z0, ..., x32, y32, z32])
X = df.to_numpy()

# Normalize coordinates between 0 and 1
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)

print("Final shape:", X_normalized.shape)  # should be (num_samples, 99)

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import csv

# Load data
df = pd.read_csv("train.csv")
df["hash_int"] = df["hash"].apply(lambda x: int(x, 16))
df["merkle_root_int"] = df["merkle_root"].apply(lambda x: int(x, 16))

# Extract and normalize features
X = df[["id", "nonce", "hash_int", "merkle_root_int", "version", "bits"]].values
y = df["nonce"].values[1:]

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X[:-1])

# Model definition
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation="relu", input_shape=[X_scaled.shape[1]]),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(1)
])

# Compile model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

# Train model
history = model.fit(X_scaled, y, epochs=50, batch_size=32, validation_split=0.2)

# Make predictions
predictions = model.predict(X_scaled)
print("Next nonce: ", predictions)

# Save results
integer_list = [int(val[0]) for val in predictions]
with open('list.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(integer_list)

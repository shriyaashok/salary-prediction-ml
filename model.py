# Importing the libraries
import numpy as np
import pandas as pd
import pickle
import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import os

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

print("=" * 60)
print("Loading and preprocessing data...")
print("=" * 60)

# Load dataset
dataset = pd.read_csv('hiring.csv')
print(f"\nDataset shape: {dataset.shape}")
print(f"\nDataset preview:\n{dataset.head()}")

# Extract features and target
X = dataset.iloc[:, :3].values  # experience, test_score, interview_score
y = dataset.iloc[:, -1].values  # salary

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Target range: ${y.min():,.0f} - ${y.max():,.0f}")

# Standardize features for CNN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape for CNN: (n_samples, n_features, 1)
X_cnn = X_scaled.reshape(X_scaled.shape[0], X_scaled.shape[1], 1)
print(f"\nCNN input shape: {X_cnn.shape}")

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')
print("\n[OK] Scaler saved as scaler.pkl")

print("\n" + "=" * 60)
print("Training 1D CNN Model...")
print("=" * 60)

# Build 1D CNN model (improved architecture for better accuracy)
def build_cnn_model(input_shape):
    model = keras.Sequential([
        layers.Conv1D(filters=32, kernel_size=2, activation='relu', input_shape=input_shape),
        layers.Dropout(0.2),
        layers.Conv1D(filters=64, kernel_size=2, activation='relu'),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dense(1)  # Output layer for regression
    ])
    # Compile without metrics to avoid serialization issues
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='mse')
    return model

# Build and train CNN
cnn_model = build_cnn_model((X_cnn.shape[1], X_cnn.shape[2]))
print("\nCNN Model Architecture:")
cnn_model.summary()

# Train CNN with improved parameters
print("\nTraining CNN model...")
# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

history = cnn_model.fit(X_cnn, y, epochs=500, batch_size=8, verbose=0, 
                        validation_split=0.2, callbacks=[early_stopping])

# Predict with CNN
y_pred_cnn = cnn_model.predict(X_cnn, verbose=0).flatten()

# Calculate CNN metrics
cnn_r2 = r2_score(y, y_pred_cnn)
cnn_mae = mean_absolute_error(y, y_pred_cnn)
cnn_mse = mean_squared_error(y, y_pred_cnn)
cnn_rmse = np.sqrt(cnn_mse)

print("\n" + "-" * 60)
print("CNN Model Metrics:")
print("-" * 60)
print(f"R² Score (Accuracy): {cnn_r2:.4f}")
print(f"MAE:  ${cnn_mae:.2f}")
print(f"MSE:  ${cnn_mse:.2f}")
print(f"RMSE: ${cnn_rmse:.2f}")

# Save CNN model
cnn_model.save('cnn_model.h5')
print("\n[OK] CNN model saved as cnn_model.h5")

print("\n" + "=" * 60)
print("Training Naive Bayes Model...")
print("=" * 60)

# For Naive Bayes regression: discretize salary into bins
# Use more bins for better granularity with larger dataset
n_bins = min(8, len(np.unique(y)))  # Use more bins for better accuracy
y_series = pd.Series(y)
y_discrete, bin_edges = pd.qcut(y_series, q=n_bins, labels=False, retbins=True, duplicates='drop')
bin_centers = [(bin_edges[i] + bin_edges[i+1]) / 2 for i in range(len(bin_edges)-1)]

# Get unique classes that exist
unique_classes = np.unique(y_discrete[~pd.isna(y_discrete)])
n_classes = len(unique_classes)

print(f"\nDiscretized salary into {n_classes} bins:")
for i, cls in enumerate(unique_classes):
    center = bin_centers[int(cls)]
    count = np.sum(y_discrete == cls)
    print(f"  Class {cls}: ${center:.0f} (count: {count})")

# Handle any NaN values (shouldn't happen, but just in case)
y_discrete = y_discrete.fillna(unique_classes[0]).astype(int).values

# Create mapping from class index to bin center
class_to_center = {int(cls): bin_centers[int(cls)] for cls in unique_classes}

# Train Naive Bayes
nb_model = GaussianNB()
nb_model.fit(X, y_discrete)

# Predict with Naive Bayes
y_pred_nb_discrete = nb_model.predict(X)
y_pred_nb_proba = nb_model.predict_proba(X)

# Convert discrete predictions back to continuous values using weighted average
# Map probabilities to bin centers based on the classes that Naive Bayes predicts
y_pred_nb_weighted = np.zeros(len(X))
for i in range(len(X)):
    weighted_sum = 0
    # Get the classes that NB can predict (these are the unique classes in training)
    nb_classes = nb_model.classes_
    for j, cls in enumerate(nb_classes):
        if int(cls) in class_to_center:
            weighted_sum += y_pred_nb_proba[i][j] * class_to_center[int(cls)]
    y_pred_nb_weighted[i] = weighted_sum

# Use weighted average for better regression performance
y_pred_nb = y_pred_nb_weighted

# Calculate Naive Bayes metrics
nb_r2 = r2_score(y, y_pred_nb)
nb_mae = mean_absolute_error(y, y_pred_nb)
nb_mse = mean_squared_error(y, y_pred_nb)
nb_rmse = np.sqrt(nb_mse)

print("\n" + "-" * 60)
print("Naive Bayes Model Metrics:")
print("-" * 60)
print(f"R² Score (Accuracy): {nb_r2:.4f}")
print(f"MAE:  ${nb_mae:.2f}")
print(f"MSE:  ${nb_mse:.2f}")
print(f"RMSE: ${nb_rmse:.2f}")

# Save Naive Bayes model and bin information
nb_data = {
    'model': nb_model,
    'bin_centers': bin_centers,
    'bin_edges': bin_edges,
    'class_to_center': class_to_center,
    'n_bins': n_bins
}
pickle.dump(nb_data, open('nb_model.pkl', 'wb'))
print("\n[OK] Naive Bayes model saved as nb_model.pkl")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print("\nSummary:")
print(f"CNN R² Score: {cnn_r2:.4f}")
print(f"Naive Bayes R² Score: {nb_r2:.4f}")
print("\nModels saved:")
print("  - cnn_model.h5")
print("  - nb_model.pkl")
print("  - scaler.pkl")

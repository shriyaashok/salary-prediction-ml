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

# Standardize features for ANN
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\nANN input shape: {X_scaled.shape}")

# Save the scaler for later use
joblib.dump(scaler, 'scaler.pkl')
print("\n[OK] Scaler saved as scaler.pkl")

print("\n" + "=" * 60)
print("Training ANN (Artificial Neural Network) Model...")
print("=" * 60)

# Build ANN model (proper architecture for tabular data)
def build_ann_model(input_dim):
    model = keras.Sequential([
        # Input layer
        layers.Dense(128, activation='relu', input_dim=input_dim),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        # Hidden layers
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        
        layers.Dense(128, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.3),
        
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.2),
        
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.1),
        
        # Output layer for regression
        layers.Dense(1, activation='linear')
    ])
    
    # Compile with appropriate optimizer and loss for regression
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    return model

# Build and train ANN
ann_model = build_ann_model(X_scaled.shape[1])
print("\nANN Model Architecture:")
ann_model.summary()

# Train ANN with improved parameters
print("\nTraining ANN model...")
# Use early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=100, restore_best_weights=True)

# Train the model
history = ann_model.fit(
    X_scaled, y, 
    epochs=1000, 
    batch_size=32, 
    verbose=1, 
    validation_split=0.2, 
    callbacks=[early_stopping]
)

# Predict with ANN
y_pred_ann = ann_model.predict(X_scaled, verbose=0).flatten()

# Calculate ANN metrics
ann_r2 = r2_score(y, y_pred_ann)
ann_mae = mean_absolute_error(y, y_pred_ann)
ann_mse = mean_squared_error(y, y_pred_ann)
ann_rmse = np.sqrt(ann_mse)

print("\n" + "-" * 60)
print("ANN Model Metrics:")
print("-" * 60)
print(f"R² Score (Accuracy): {ann_r2:.4f}")
print(f"MAE:  ${ann_mae:.2f}")
print(f"MSE:  ${ann_mse:.2f}")
print(f"RMSE: ${ann_rmse:.2f}")

# Save ANN model
ann_model.save('ann_model.h5')
print("\n[OK] ANN model saved as ann_model.h5")

print("\n" + "=" * 60)
print("Training Enhanced Naive Bayes Model...")
print("=" * 60)

# Enhanced Naive Bayes with multiple improvements
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import QuantileTransformer

# 1. Feature Engineering - Add polynomial features for better relationships
X_enhanced = X.copy()

# Add interaction features
experience_col = X[:, 0]
test_score_col = X[:, 1] 
interview_score_col = X[:, 2]

# Add meaningful feature combinations
avg_score = (test_score_col + interview_score_col) / 2
score_diff = np.abs(test_score_col - interview_score_col)
exp_score_ratio = experience_col / (avg_score + 1e-8)

# Stack new features
X_enhanced = np.column_stack([
    X,  # Original features
    avg_score,  # Average of test and interview scores
    score_diff,  # Difference between scores (consistency measure)
    exp_score_ratio,  # Experience to score ratio
    experience_col ** 0.5,  # Square root of experience (diminishing returns)
    avg_score ** 2  # Squared average score (non-linear effect)
])

print(f"Enhanced features shape: {X_enhanced.shape}")

# 2. Better discretization strategy - use quantile-based binning with more bins
n_bins = min(12, len(np.unique(y)))  # Increased bins for better granularity
y_series = pd.Series(y)

# Use quantile-based binning for more balanced classes
y_discrete, bin_edges = pd.qcut(y_series, q=n_bins, labels=False, retbins=True, duplicates='drop')

# Calculate bin centers more accurately
bin_centers = []
for i in range(len(bin_edges)-1):
    # Use actual data points in each bin for more accurate centers
    bin_mask = (y >= bin_edges[i]) & (y <= bin_edges[i+1])
    if np.any(bin_mask):
        bin_centers.append(np.mean(y[bin_mask]))
    else:
        bin_centers.append((bin_edges[i] + bin_edges[i+1]) / 2)

# Get unique classes that exist
unique_classes = np.unique(y_discrete[~pd.isna(y_discrete)])
n_classes = len(unique_classes)

print(f"\nDiscretized salary into {n_classes} bins:")
for i, cls in enumerate(unique_classes):
    if int(cls) < len(bin_centers):
        center = bin_centers[int(cls)]
        count = np.sum(y_discrete == cls)
        print(f"  Class {cls}: ${center:.0f} (count: {count})")

# Handle any NaN values
y_discrete = y_discrete.fillna(unique_classes[0]).astype(int).values

# Create mapping from class index to bin center
class_to_center = {int(cls): bin_centers[int(cls)] if int(cls) < len(bin_centers) else bin_centers[-1] 
                   for cls in unique_classes}

# 3. Use Bagging with multiple Naive Bayes models for better performance
print("\nTraining ensemble of Naive Bayes models...")
nb_ensemble = BaggingClassifier(
    estimator=GaussianNB(),
    n_estimators=15,  # Multiple models for better accuracy
    random_state=42,
    bootstrap=True,
    bootstrap_features=False
)

# Train ensemble
nb_ensemble.fit(X_enhanced, y_discrete)

# Also train a single model for comparison
nb_single = GaussianNB()
nb_single.fit(X_enhanced, y_discrete)

# 4. Enhanced prediction with ensemble
y_pred_nb_proba_ensemble = nb_ensemble.predict_proba(X_enhanced)
y_pred_nb_proba_single = nb_single.predict_proba(X_enhanced)

# Combine ensemble and single model predictions (weighted average)
ensemble_weight = 0.7
single_weight = 0.3
y_pred_nb_proba_combined = (ensemble_weight * y_pred_nb_proba_ensemble + 
                           single_weight * y_pred_nb_proba_single)

# Convert to continuous predictions using improved weighted average
y_pred_nb_weighted = np.zeros(len(X_enhanced))
ensemble_classes = nb_ensemble.classes_

for i in range(len(X_enhanced)):
    weighted_sum = 0
    total_weight = 0
    
    for j, cls in enumerate(ensemble_classes):
        if int(cls) in class_to_center:
            prob = y_pred_nb_proba_combined[i][j]
            center = class_to_center[int(cls)]
            weighted_sum += prob * center
            total_weight += prob
    
    if total_weight > 0:
        y_pred_nb_weighted[i] = weighted_sum / total_weight
    else:
        y_pred_nb_weighted[i] = np.mean(list(class_to_center.values()))

# Use the enhanced prediction
y_pred_nb = y_pred_nb_weighted

# Store both models (use ensemble as primary)
nb_model = nb_ensemble

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

# Save Enhanced Naive Bayes model and bin information
nb_data = {
    'model': nb_model,  # Ensemble model
    'single_model': nb_single,  # Single model for comparison
    'bin_centers': bin_centers,
    'bin_edges': bin_edges,
    'class_to_center': class_to_center,
    'n_bins': n_bins,
    'ensemble_weight': ensemble_weight,
    'single_weight': single_weight,
    'feature_engineering': True  # Flag to indicate enhanced features are used
}
pickle.dump(nb_data, open('nb_model.pkl', 'wb'))
print("\n[OK] Enhanced Naive Bayes model saved as nb_model.pkl")

print("\n" + "=" * 60)
print("Training Complete!")
print("=" * 60)
print("\nSummary:")
print(f"ANN R² Score: {ann_r2:.4f}")
print(f"Naive Bayes R² Score: {nb_r2:.4f}")
print("\nModels saved:")
print("  - ann_model.h5")
print("  - nb_model.pkl")
print("  - scaler.pkl")

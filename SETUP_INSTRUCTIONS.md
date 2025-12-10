# Salary Prediction ML Project - Setup Instructions

## Project Overview
This project predicts employee salary using two machine learning models:
1. **1D CNN Model** (Keras/TensorFlow) - Deep learning approach
2. **Naive Bayes Model** (scikit-learn) - Probabilistic classification converted to regression

## Installation

1. Install required dependencies:
```bash
pip install -r requirements.txt
```

## Running the Project

### Step 1: Train the Models
First, train both models and save them:
```bash
python model.py
```

This will:
- Load `hiring.csv`
- Train CNN model and save as `cnn_model.h5`
- Train Naive Bayes model and save as `nb_model.pkl`
- Save the scaler as `scaler.pkl`
- Print metrics (R², MAE, MSE, RMSE) for both models

### Step 2: Run the Flask Application
Start the Flask server:
```bash
python app.py
```

Or alternatively:
```bash
python application.py
```

The server will start on `http://127.0.0.1:5000`

### Step 3: Use the Web Interface
1. Open your browser and navigate to `http://127.0.0.1:5000`
2. Enter the following values:
   - **Experience** (years)
   - **Test Score**
   - **Interview Score**
3. Click "Predict Salary"
4. View predictions from both models along with their accuracy scores

## API Endpoint

You can also use the JSON API endpoint:

**POST** `/api/predict`

Request body:
```json
{
  "experience": 5,
  "test_score": 8,
  "interview_score": 9
}
```

Response:
```json
{
  "cnn_salary_prediction": 62500.00,
  "cnn_accuracy": 0.9234,
  "naive_bayes_salary_prediction": 63000.00,
  "naive_bayes_accuracy": 0.9306
}
```

## Model Details

### CNN Model Architecture
- Input: (3, 1) - reshaped from 3 features
- Conv1D layer (16 filters)
- Flatten layer
- Dense layers (32, 16 neurons)
- Output: Single value (salary prediction)

### Naive Bayes Model
- Uses GaussianNB classifier
- Discretizes salary into bins
- Converts classification output back to continuous regression values using weighted probabilities

## Files Structure
- `model.py` - Trains both models and saves them
- `app.py` / `application.py` - Flask backend application
- `templates/index.html` - Web interface
- `static/css/style.css` - Styling
- `hiring.csv` - Training dataset
- `cnn_model.h5` - Saved CNN model
- `nb_model.pkl` - Saved Naive Bayes model
- `scaler.pkl` - Feature scaler for CNN

## Notes
- The dataset is very small (8 samples), so CNN performance may vary
- Naive Bayes typically performs better on small datasets
- Both models are retrained each time you run `model.py`


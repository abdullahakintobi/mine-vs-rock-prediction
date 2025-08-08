# --- This script trains the Logistic Regression model and saves it to a file ---

# Import necessary libraries
import os

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# --- Data Preparation and Training ---

# Load dataset.
print("---" * 15)
print("Loading data...")
try:
    sonar_data = pd.read_csv("../data/sonar_data.csv", header=None)
except FileNotFoundError:
    print(
        "Error: 'sonar_data.csv' not found. Please ensure the file is in the data directory."
    )
    exit()

# Separate the data into features (X) and target labels (Y)
X = sonar_data.drop(60, axis=1)
Y = sonar_data[60]

# Split the dataset into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1
)

# --- Model Training ---

# Initialize and train the Logistic Regression model
print("Training the model...")
model = LogisticRegression()
model.fit(X_train, Y_train)
print("Model training complete.")

# --- Model Evaluation ---

# Check accuracy on the training data
print("---" * 15)
X_train_prediction = model.predict(X_train)
train_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f"Accuracy on train data: {train_data_accuracy * 100:.2f}%")

# Check accuracy on the test data
X_test_prediction = model.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Accuracy on test data: {test_data_accuracy * 100:.2f}%")
print("---" * 15)

# --- Save the Trained Model ---

# Define the directory to save the model
model_dir = "../model"
# Create the directory if it doesn't exist
os.makedirs(model_dir, exist_ok=True)
# Save the trained model to the specified path
model_path = os.path.join(model_dir, "sonar_model.pkl")
joblib.dump(model, model_path)
print(f"Trained model saved to '{model_path}'.")
print("---" * 15)

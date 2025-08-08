# This script loads the saved model and uses it to make a prediction on new data.

# Import necessary libraries
import joblib
import numpy as np

# --- Load the Saved Model ---

# Define the path to the saved model
model_path = "../model/sonar_model.pkl"

# Load the trained model from the 'sonar_model.pkl' file
print("---" * 12)
print("Loading model...")
try:
    model = joblib.load(model_path)
    print("Model loaded successfully.")
except FileNotFoundError:
    print(
        "Error: 'sonar_model.pkl' not found in the '../model' directory. Please run 'train_model.py' first."
    )
    exit()

# --- Prepare New Input Data ---

# Insert the data you want to predict
input_data = (
    0.0270, 0.0163, 0.0346, 0.0216, 0.0637, 0.1278, 0.1873, 0.2280, 0.3113, 0.3807,
    0.3703, 0.4075, 0.3204, 0.2312, 0.2038, 0.2520, 0.2687, 0.2882, 0.3168, 0.2792,
    0.2818, 0.3831, 0.4659, 0.5057, 0.5369, 0.5960, 0.6558, 0.7011, 0.7301, 0.7180,
    0.7061, 0.6723, 0.6405, 0.5591, 0.5186, 0.4571, 0.3957, 0.3664, 0.3142, 0.2384,
    0.1983, 0.1420, 0.0898, 0.0768, 0.0460, 0.0292, 0.0181, 0.0076, 0.0063, 0.0118,
    0.0102, 0.0031, 0.0075, 0.0115, 0.0160, 0.0094, 0.0064, 0.0089, 0.0069, 0.0121,
)

# Convert the input data tuple to a NumPy array
input_data_array = np.asarray(input_data)

# Reshape the array to indicate that we're predicting for a single instance
input_data_reshape = input_data_array.reshape(1, -1)

# --- Make Prediction ---

# Use the loaded model to make a prediction
prediction = model.predict(input_data_reshape)

# Print the prediction result
print("---" * 12)
if prediction[0] == "R":
    print("Prediction: The object is a Rock :)")
else:
    print("Prediction: The object is a Mine :o")
print("---" * 12)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
import joblib

# Load the dataset
data = pd.read_csv("11_26_24_training.csv") # TODO: Replace with your own training data

# Separate features and labels
X = data.drop("label", axis=1)  # Features: landmark coordinates
y = data["label"]               # Labels: good (1) or bad (0) posture

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (optional but can improve model performance)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Build the model
# NOTE: 14 here represents the x and y values for the 7 body parts that we're tracking with mediapipe.
#       hidden layers can be modified to alleviate over/under fitting (e.g., by adjusting the number of neurons or layers)
model = models.Sequential([
    layers.Input(shape=(14,)),            # Input layer for 14 landmark features
    layers.Dense(32, activation='relu'),  # Hidden layer with 32 neurons
    layers.Dense(16, activation='relu'),  # Hidden layer with 16 neurons
    layers.Dense(1, activation='sigmoid') # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
# NOTE: During model iteration with varying data sizes, I often change epochs to try to get the best accuracy.
#       I usually test with epochs ranging from 30-100
history = model.fit(X_train, y_train, epochs=40, batch_size=8, validation_data=(X_test, y_test))

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_accuracy:.2f}")

# Save the model to a file
model.save("posture_model.h5")
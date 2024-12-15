import joblib
import cv2
import os
import time
import pandas as pd
import random
import numpy as np
import mediapipe as mp
import punishment

from tensorflow.keras.models import load_model


# Load the previously saved model
# TODO: Replace with your own training model and scaler generated from build_model.py
model = load_model("posture_model_11_26.h5")
scaler = joblib.load("scaler_11_26.pkl")

# Load camera
output_dir = "ofp"
os.makedirs(output_dir, exist_ok=True)
# Open the default camera (camera index 0)
camera = cv2.VideoCapture(0)
# Set the camera resolution to 4K (3840x2160)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

# Initialize Mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

def classify_data(data):
    # Load and preprocess the new data
    print("Analyzing posture...")
    data_array = np.array(data)

    # Exclude the label (if included) for classification
    data_no_labels = data_array[:, :-1] if data_array.shape[1] > 14 else data_array  # Remove last column if it's the label
    # Scale the data using the pre-trained scaler
    new_data_scaled = scaler.transform(data_no_labels)

    # Classify the new data
    predictions = model.predict(new_data_scaled)
    classified_labels = (predictions > 0.5).astype(int)

    # Save the results
    print("Predicted Labels:", classified_labels)
    print("Classification completed.")

    return classified_labels[0][0]

def capture_image():
    if not camera.isOpened():
        print("Error: Could not access the camera.")
        exit()

    ret, frame = camera.read()
    print("Capturing image...")
    if not ret:
        print("Error: Could not read frame from the camera.")
        exit()

    return frame

def preprocess_data(image):
    # run image through MP
    data = []
    if image is not None:
        print("Preprocessing data with MP...")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Perform pose detection
        result = pose.process(rgb_image)
        # Check if landmarks are detected
        if result.pose_landmarks:
            landmarks = result.pose_landmarks.landmark
            # Select only the left-side and key center landmarks
            relevant_landmarks = [
                mp_pose.PoseLandmark.NOSE,           # Head posture (if visible)
                mp_pose.PoseLandmark.LEFT_SHOULDER,  
                mp_pose.PoseLandmark.LEFT_ELBOW,
                mp_pose.PoseLandmark.LEFT_WRIST,
                mp_pose.PoseLandmark.LEFT_HIP,
                mp_pose.PoseLandmark.LEFT_KNEE,
                mp_pose.PoseLandmark.LEFT_ANKLE
            ]
            # Extract the (x, y) coordinates for these landmarks only
            pose_landmarks_xy = [(landmarks[landmark.value].x, landmarks[landmark.value].y) for landmark in relevant_landmarks]
            # Convert to numpy array and flatten
            pose_landmarks_array_xy = np.array(pose_landmarks_xy).flatten()
            # Append the flattened pose landmarks and label to the data
            data.append(pose_landmarks_array_xy)
            
            df = pd.DataFrame(data, columns=scaler.feature_names_in_)
            return df
    return None


def save_image(image, classification, index):
    label = 't' if classification == 1 else 'f'
    filename = f"pos{index}_{label}.jpg"
    print(f"Saving image to {output_dir}/{filename}")
    image_path = os.path.join(output_dir, filename)
    cv2.imwrite(image_path, image)

def do_punishment(bad_count):
    if bad_count <= 1: # incase you forget to increment before sending lets do <= 1
        severity : punishment.Severity = punishment.Severity.LOW
    elif bad_count < 3:
        severity : punishment.Severity = punishment.Severity.MID
    else:
        severity : punishment.Severity = punishment.Severity.HIGH
    punishment.execute(severity)

if __name__ == '__main__':
    count = 0
    bad_count = 0
    try:
        while True:
            delay = random.uniform(0, 5)
            print(f"Waiting for {delay:.2f} seconds...")
            time.sleep(delay)  # Wait for the random duration
            image = capture_image()
            data = preprocess_data(image)
            classification = classify_data(data)
            save_image(image, classification, count)
            if classification == 0:
                print("Bad posture detected!")
                print("Sending punishment request...")
                do_punishment(bad_count)
                bad_count += 1
            else:
                bad_count = 0
                print("Good posture!")
            count += 1
    except KeyboardInterrupt:
        print("Stopping OFP")
        camera.release()
        cv2.destroyAllWindows()
import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os

# Initialize Mediapipe pose detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Specify the directory containing images
processed_data_file_name = '11_26_24_training' # training data
data = []
# My images were partitioned per day so thats why it takes in an array of folders instead of just one path
image_folders = [] # TODO: Add folder paths where your images are stored.

for image_folder in image_folders:
# Loop through all files in the specified directory
    for filename in os.listdir(image_folder):
        # Check if the file is an image (you can add more extensions if needed)
        if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
            # Construct full file path
            image_path = os.path.join(image_folder, filename)
            
            # Read the image using OpenCV
            image = cv2.imread(image_path)
            
            # Check if the image was loaded successfully
            if image is not None:
                # Convert the image to RGB as mediapipe needs RGB input
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Perform pose detection
                result = pose.process(rgb_image)

                # Check if landmarks are detected
                if result.pose_landmarks:
                    mp_drawing.draw_landmarks(image, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                    landmarks = result.pose_landmarks.landmark
                    """
                    NOTE: I figured that the optimal angle to check for bad posture was on the side.

                    Were only tracking LEFT body parts with the assumption that the RIGHT
                    body parts would just be mirrored. The main reason we dont track RIGHT body parts is
                    because it will most likely be out of view from the cameras perspective. 
                    """
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
                    pose_landmarks_array_xy = np.array(pose_landmarks_xy).flatten()  # Shape: (7*2,) or (14,)
                    print("Filtered pose landmarks (x, y) array:", pose_landmarks_array_xy)
                
                    # Display the annotated image
                    # cv2.imshow('Posture Detection', image)
                    # cv2.waitKey(0)  # Wait for a key press to close the window

                    # Example posture label (0 or 1), replace with your actual label
                    # NOTE: I manually labeled each image, appending either _t or _f to label good and bad posture
                    print(image_path.split('_')[1][0])
                    posture_label = 1 if image_path.split('_')[1][0] == 't' else 0

                    # Append the flattened pose landmarks and label to the data
                    data.append(np.append(pose_landmarks_array_xy, posture_label))
            else:
                print("Could not read image from the specified path.")


columns = [f"landmark_{i}_x" if i % 2 == 0 else f"landmark_{i//2}_y" for i in range(14)]
columns.append("label")  # Add label column
df = pd.DataFrame(data, columns=columns)
df['label'] = df['label'].astype(int)

# Save the data to a CSV file
df.to_csv(f"{processed_data_file_name}.csv", index=False)
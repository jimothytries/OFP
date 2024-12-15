import cv2
import os
import time

# Create a directory to save the captured images
output_dir = "training-images"
os.makedirs(output_dir, exist_ok=True)

# Open the default camera (camera index 0)
camera = cv2.VideoCapture(0)
# Set the camera resolution to 4K (3840x2160)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)
# Check if the camera is opened successfully
if not camera.isOpened():
    print("Error: Could not access the camera.")
    exit()

print("Press Ctrl+C to stop capturing images.")

count = 0
target_count = 100 # stop after taking this number of images

try:
    while True:
        ret, frame = camera.read()

        if not ret:
            print("Error: Could not read frame from the camera.")
            break

        # Save the frame as an image file
        image_path = os.path.join(output_dir, f"test{count}_t.jpg")
        cv2.imwrite(image_path, frame)

        print(f"Captured and saved: {image_path}")

        count += 1
        time.sleep(5) # NOTE: Adjust the time interval between taking images

        if count == target_count:
            break

except KeyboardInterrupt:
    print("\nStopping image capture.")

finally:
    # Release the camera and close all OpenCV windows
    camera.release()
    cv2.destroyAllWindows()
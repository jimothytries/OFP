# OFP
A Python-based project that utilizes machine learning libraries such as **MediaPipe** and **TensorFlow** to classify posture in real-time. The goal is to punish the user when their posture is bad.

# Project Workflow
1. **capture_images.py**
    - Capture images of your posture.
    - Make sure to manually label each image with `_t` (good posture) or `_f` (bad posture).
    - Used for generating images to feed into **#2**.
   
2. **build_training_data.py**
    - Uses **MediaPipe** to extract x and y coordinates of relevant features.
    - Creates a labeled CSV file for training data.
   
3. **build_model.py**
    - Builds the model using the training data generated in **#2**.
   
4. **ofp.py**
    - Uses the model from **#3** to predict whether the posture in an image is good or bad.
5. **punishment.py**
    - Meant to be an interface to support different types of punishments but currently only supports one, which is a call to an IOT device

# Setup
### Create Virtual Env:
```powershell
python -m venv <name_of_env>
```

### Activate Virtual Env:
```powershell
./<name_of_env>/Scripts/activate.ps1
```

### Install dependencies
```powershell
pip install -r requirements.txt
```

# Notes
* The training data I have might be specific to my configuration (eg. body part measurements, camera distance, camera resolution), so using the same training data(`_11_26_24_training.csv`) or same model(`_posture_model_11_26.h5`) for your setup might not work. I do recommend at least 100 images to start with, but for more accurate predictions around 500+ is good
* There are TODO comments in the code that you'll need to replace to adapt it to your own setup.
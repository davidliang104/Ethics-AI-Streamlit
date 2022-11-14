import streamlit as st
import os
import sys
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import subprocess
import torch

# from keras.utils import to_categorical
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# Load model
model = load_model(os.path.join('models', 'model_cls_810_28_700_gb_imb.h5'))
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

image_dir = 'images'
det_dir = os.path.join('yolov5', 'runs', 'detect', 'exp')
label_dir = os.path.join(det_dir, 'labels')

def silent_remove(filename):
    if os.path.exists(filename) and os.path.isfile(filename):
        os.remove(filename)

def empty_dir(dir):
    for file in os.listdir(dir):
        if file != '.gitkeep':
            silent_remove(os.path.join(dir, file))

if 'key' not in st.session_state:
    st.session_state['key'] = 'first'
    # Empty all directories at start of session
    empty_dir(image_dir)
    empty_dir(det_dir)
    empty_dir(label_dir)


# @st.cache
# def clear_all_dir():
#     # Empty all directories at start of session
#     empty_dir(image_dir)
#     empty_dir(det_dir)
#     empty_dir(label_dir)

def image_prep_cls(image):
  # Image preprocessing
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to greyscale
  image = cv2.resize(image, (700,700)) # Resizing
  image = cv2.bilateralFilter(image,15,75,75) # Apply bilateral filter for blurring
  image = image / 255.0 # Normalizing
  return image

def predict_cls(image_path):
    image = cv2.imread(image_path)
    image_prepped = np.array(image_prep_cls(image)) # Preprocess image
    image_prepped = np.expand_dims(image_prepped, axis=0)

    prediction = model.predict(image_prepped)
    output = np.argmax(prediction, axis=1) # Undo one-hot encoding

    label_names = {0:'sexual harassment', 1:'sexual abuse', 2:'sexual violence'}
    output = np.array([label_names[x] for x in output]) # Convert label to words
    output_v = output[0].capitalize() # Get value of only element in the array

    return output_v

# def img_prep_box(image):
#     # Image preprocessing
#     image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to greyscale
#     image = cv2.bilateralFilter(image,15,75,75) # Apply bilateral filter for blurring
#     # No need to normalize because the YOLO model already does that
#     return image

def predict_box(image_dir):
    # Image augmentation
    # img_list = os.listdir(image_dir)
    # for image in img_list:
    #     img_path = os.path.join(image_dir, image)
    #     img_array = cv2.imread(img_path)
    #     img_array = img_prep_box(img_array)
    #     cv2.imwrite(img_path, img_array)

    # Run prediction
    subprocess.run([f"{sys.executable}", 'yolov5/detect.py', '--weights', os.path.join('models','best_iaug720.pt'), '--img', '720', '--conf', '0.4', '--source', image_dir, '--save-txt', '--save-conf', '--exist-ok'])

def save_uploaded_file(uploaded_files):
    file_names = []
    for file in uploaded_files:
        with open(os.path.join('images',file.name),'wb') as f:
            f.write(file.getbuffer())
        file_names.append(file.name)

    # Remove images not in uploaded files anymore
    for image in os.listdir(image_dir):
        print("File name:",image)
        if image not in file_names and image != '.gitkeep':
            silent_remove(os.path.join(image_dir, image))
            silent_remove(os.path.join(det_dir, image))
            image_name = image.split(".")
            silent_remove(os.path.join(label_dir, image_name[0]+'.txt'))

    return 1

uploaded_files = st.file_uploader("Upload image", accept_multiple_files=True, type=["png","jpg","jpeg"])

if uploaded_files is not None:
    if len(uploaded_files)>0 and save_uploaded_file(uploaded_files):        
        # Predict all bounding boxes
        predict_box(image_dir)

        for image in os.listdir(image_dir):
            box_img_path = os.path.join(det_dir, image)
            box_pred = cv2.imread(box_img_path)
            display_img = cv2.cvtColor(box_pred, cv2.COLOR_RGB2BGR)
            st.image(display_img)

            image_path = os.path.join(image_dir, image)
            # Get classification
            cls_prediction = predict_cls(image_path)
            st.success(cls_prediction) # Display prediction
            st.text("")
            st.text("")
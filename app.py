import streamlit as st

import os
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
model = load_model('model_cls_810_28_700_gb.h5')
print(f"Setup complete. Using torch {torch.__version__} ({torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'})")

image_dir = 'images'
#'yolov5\\runs\\detect\\exp'
det_dir = os.path.join('yolov5', 'runs', 'detect', 'exp')
label_dir = os.path.join(det_dir, 'labels')

def image_prep_cls(image):
  # Image preprocessing
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to greyscale
  image = cv2.resize(image, (700,700)) # Resizing
  image = cv2.bilateralFilter(image,15,75,75) # Apply bilateral filter for blurring
  image = image / 255.0 # Normalizing
  return image

# def predict_cls(img_dir):
#     x_test = []

#     for image_name in os.listdir(img_dir):
#         image = cv2.imread(os.path.join(img_dir,image_name))
#         image = image_prep_cls(image)
#         x_test.append(image)
#     x_test = np.array(x_test)

#     prediction = model.predict(x_test)
#     y_pred = np.argmax(prediction, axis=1) # Undo one-hot encoding

#     label_names = {0:'sexual harassment', 1:'sexual abuse', 2:'sexual violence'}
#     y_pred = np.array([label_names[x] for x in y_pred])

#     return y_pred

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

def img_prep_box(image):
    # Image preprocessing
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) # Convert to greyscale
    image = cv2.bilateralFilter(image,15,75,75) # Apply bilateral filter for blurring
    # No need to normalize because the YOLO model already does that
    return image

def predict_box(image_dir):
    # image = cv2.imread(image_path)
    # Run prediction
    subprocess.run(['python', 'yolov5/detect.py', '--weights', 'best_iaug720.pt', '--img', '720', '--conf', '0.4', '--source', image_dir, '--save-txt', '--save-conf', '--exist-ok'])


def save_uploaded_file(uploaded_files):
    file_names = []
    for file in uploaded_files:
        with open(os.path.join('images',file.name),'wb') as f:
            f.write(file.getbuffer())
        file_names.append(file.name)

    # Remove images not in uploaded files anymore
    for image in os.listdir(image_dir):
        if image not in file_names:
            os.remove(os.path.join(image_dir, image))
            os.remove(os.path.join(det_dir, image))
            os.remove(os.path.join(label_dir, image[:-4]+'.txt'))

    return 1

uploaded_files = st.file_uploader("Upload image", accept_multiple_files=True, type=["png","jpg","jpeg"])

if uploaded_files is not None:
    if save_uploaded_file(uploaded_files):        
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

        # for file in uploaded_files:
        #     image_path = os.path.join('images',file.name)

        #     # Display the image
        #     display_image = cv2.imread(image_path)
        #     display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
        #     st.image(display_image)

        #     predict_box(image_path)

        #     # # Get classification
        #     # cls_prediction = predict_cls(image_path)
        #     # st.success(cls_prediction) # Display prediction

        #     # Delete uploaded image after prediction
        #     os.remove(image_path) 
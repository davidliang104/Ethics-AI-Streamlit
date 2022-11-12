import streamlit as st

import os
import numpy as np
import cv2
from tensorflow.keras.models import load_model
# from keras.utils import to_categorical
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt

# Load model
model = load_model('model_cls_810_28_700_gb.h5')

def image_prep_cls(image):
  # Image preprocessing
  image = cv2.resize(image, (720,720)) # Resizing to 720x720
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

def save_uploaded_file(uploaded_file):
    for file in uploaded_files:
        with open(os.path.join('images',file.name),'wb') as f:
            f.write(file.getbuffer())
    return 1

uploaded_files = st.file_uploader("Upload image", accept_multiple_files=True, type=["png","jpg","jpeg"])

if uploaded_files is not None:
    if save_uploaded_file(uploaded_files):
        for file in uploaded_files:
            image_path = os.path.join('images',file.name)

            # Display the image
            display_image = cv2.imread(image_path)
            display_image = cv2.cvtColor(display_image, cv2.COLOR_RGB2BGR)
            st.image(display_image)

            # Get classification
            cls_prediction = predict_cls(image_path)
            st.success(cls_prediction) # Display prediction

            # Delete uploaded image after prediction
            os.remove(image_path) 
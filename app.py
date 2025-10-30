import streamlit as st
import os
from PIL import Image
import numpy as np
import cv2
from keras.models import load_model
from util import set_background  

bg = cv2.imread('./bgs/bg5.png')

blurred_bg = cv2.GaussianBlur(bg, (15, 15), 0)

cv2.imwrite('./bgs/bg5_blur.png', blurred_bg)

# Set background
set_background('./bgs/bg5_blur.png')

# Set title and header
st.title('🩺 Pneumonia Classifier Application')
st.header('Upload a Chest X-ray Image')

# Upload file
file = st.file_uploader('', type=['jpeg', 'jpg', 'png'])

# Load model
model_path = os.path.join(os.path.dirname(__file__), 'model', 'pneumonia_classifier.keras')
model = load_model(model_path)


class_names = ['NORMAL', 'PNEUMONIA']

# Display image and classify
if file is not None:
    image = Image.open(file).convert('L')  
    st.image(image, caption="Uploaded X-ray", use_container_width=True)


    img = np.array(image)
    img = cv2.resize(img, (128, 128))
    img = img / 255.0
    img = np.expand_dims(img, axis=(0, -1))  

    # Predict
    prediction = model.predict(img)
    class_idx = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display results
    st.write("## Prediction: {}".format(class_names[class_idx]))
    st.write("### Confidence: {:.2f}%".format(confidence * 100))

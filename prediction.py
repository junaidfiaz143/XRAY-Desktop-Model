from tensorflow.keras.models import load_model
import cv2
import numpy as np
import os

# load model
model = load_model("model/chest_xray_model.h5")

os.system("cls")

# input_name = "normal.jpeg"
input_name = "pneumonia.jpeg"

# load input image
input_image = cv2.imread(input_name)
# resize input image on which model is trained
input_image = cv2.resize(input_image, (224, 224))

# expand input dimension to add dimension for batch_size
input_image = np.expand_dims(input_image, axis=0)

print("Input Image Name:", input_name)
print("Input Image Dimension: ", input_image.shape)

# get predictions from model
predictions = model.predict(input_image)[0]

print("{'NORMAL': 0, 'PNEUMONIA': 1}")
print(predictions)
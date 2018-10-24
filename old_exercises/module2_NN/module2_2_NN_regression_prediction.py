# Module 2: Neural Network
# Load Model and Predict

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import pandas as pd
from keras.models import load_model

# Load the model
model = load_model('trained_model2.h5')

# Read the new products
X = pd.read_csv('../data/proposed_new_product.csv')

# Prediction
prediction = model.predict(X)
prediction = (prediction + 0.153415)/0.0000042367
print(prediction)

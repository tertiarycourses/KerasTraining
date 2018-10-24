# Module 5: Transfer Learning
# InceptionV3 Prediction

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input,decode_predictions
from keras.preprocessing import image
import numpy as np

# Step 1: Preprocess data

# Load the image file, resizing it to 299x299 pixels (required by this model)
img = image.load_img("../images/bay-1366.jpg", target_size=(299, 299))

# Convert the image to a numpy array
x = image.img_to_array(img)

# Add a forth dimension since Keras expects a list of images
x = np.expand_dims(x, axis=0)

# Step 2: Load Pre-trained Model

# Load Keras' InceptionV3 model that was pre-trained against the ImageNet database
#model = InceptionV3(weights='imagenet', include_top=True, input_tensor=None, input_shape=None)
#print(model.summary())
model = InceptionV3()

# Scale the input image to the range used in the trained network
x = preprocess_input(x)

# Run the image through the deep neural network to make a prediction
predictions = model.predict(x)

# Look up the names of the predicted classes. Index zero is the results for the first image.
predicted_classes = decode_predictions(predictions, top=5)

print("This is an image of:")

for imagenet_id, name, likelihood in predicted_classes[0]:
    print(" - {}: {:2f} likelihood".format(name, likelihood*100.0))



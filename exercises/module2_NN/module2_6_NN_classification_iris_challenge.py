# Module 2 Neural Network
# Challenge: NN Classification Model for iris dataaset


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Parameters
n_features = 4
n_classes = 3
learning_rate = 0.08
training_epochs = 50
logdir = '/tmp/iris/1'

import keras
from keras.layers import Dense, Activation
from keras.models import Sequential

# Step 1: Preprocess Data
import numpy as np
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data
target = iris.target

# Convert the label into one-hot vector
num_labels = len(np.unique(target))
Y = np.eye(num_labels)[target]

# Split the data into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.33, random_state=42)

# Step 2: Define Model
L1 = 100
L2 = 40
L3 = 20
model = Sequential()
model.add(Dense(L1, input_dim=n_features, activation='relu'))
model.add(Dense(L2, activation='relu'))
model.add(Dense(L3, activation='relu'))
model.add(Dense(n_classes, activation='softmax'))

# Step 3: Compile Model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

logger = keras.callbacks.TensorBoard(
    log_dir='temp/keras',
    histogram_freq=4,
    write_graph=True
)

# Step 4: Train Mdodel
model.fit(X_train,y_train, epochs=training_epochs, verbose=2,callbacks = [logger])

# Step 5: Evaluatr Model 
score = model.evaluate(X_test, y_test)
print("\nTraining Accuracy = ",score[1],"Loss",score[0])

# Step 6: Save Model
model.save("trained_model_iris.h5")
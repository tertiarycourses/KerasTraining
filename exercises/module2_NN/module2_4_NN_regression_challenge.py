# Module 2: Neural Network
# Challenge: Neural Network Model for Regression

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import *

# Step 1: Preprocess Data
X = np.linspace(0,10,200)
Y = -X + np.random.randn(len(X))

# plt.scatter(X,Y)
# plt.show()

# Step 2: Define Model
model = Sequential()
model.add(Dense(50, input_dim=1, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='linear'))

# Step 3: Compile Model
model.compile(loss='mean_squared_error', optimizer='sgd')

# Step 4: Train Model
model.fit(X,Y,epochs=50)

# Step 5: Evaluate Mode
Yhat = model.predict(X)
plt.scatter(X,Y)
plt.plot(X,Yhat,'r')
plt.show()

# Step 6: Save Model
model.save("test.h5")


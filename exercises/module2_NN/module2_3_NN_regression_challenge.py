# Module 2: Neural Network
# Challenge: Neural Network Model for Regression

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense

# Step 1 Load the Model
X = np.linspace(0,10,200)
y = -X + np.random.randn(len(X))

# Step 2 Build the Model
model = Sequential()
model.add(Dense(10,input_dim=1,activation='relu'))
model.add(Dense(1,activation='linear'))

# Step 3 Compile the Model
model.compile(loss="mean_squared_error",optimizer='sgd')

# Step 4: Train the Model
model.fit(X,y,epochs=20)

# Step 5: Evaluate the Model
yhat = model.predict(X)

plt.scatter(X,y)
plt.plot(X,yhat,'r')
plt.show()
# Module 2: Neural Network
# Neural Network Model for Regression

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import pandas as pd
from keras.models import Sequential
from keras.layers import *

# Step 1: Preprocess Data
training_data_df = pd.read_csv("sales_data_training_scaled.csv")
X = training_data_df.drop('total_earnings', axis=1).values
Y = training_data_df[['total_earnings']].values

test_data_df = pd.read_csv("sales_data_testing_scaled.csv")
X_test = test_data_df.drop('total_earnings', axis=1).values
Y_test = test_data_df[['total_earnings']].values

# Step 2: Define Model
model = Sequential()
model.add(Dense(50, input_dim=9, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1, activation='linear'))
#print(model.summary())

# Step 3: Compile Model
model.compile(loss='mean_squared_error', optimizer='adam')

# Step 4: Train Model
model.fit(X,Y,epochs=50,shuffle=True)

# Step 5: Evaluate Model
test_error_rate = model.evaluate(X_test, Y_test)
print("The mean squared error (MSE) for the test data set is: {}".format(test_error_rate))

# Step 6: Save Model
model.save("trained_model.h5")


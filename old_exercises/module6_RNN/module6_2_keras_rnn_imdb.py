# Module 6 RNN
# IMDB

import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense

# Step 1: Preprocess the data
from keras.datasets import imdb
(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=20000)

# print(X_train)
# print(y_train)

X_train = sequence.pad_sequences(X_train,maxlen=80)
X_test = sequence.pad_sequences(X_test,maxlen=80)

# Step 2: Build the Model
model = Sequential()
model.add(Embedding(20000,128))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))

# Step 3: Compile the Model
model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

# Step 4: Train the Model
model.fit(X_train,y_train,epochs=2)

# Step 5: Evalaute the Model
loss,accuracy = model.evaluate(X_test,y_test)
print('Loss = ',loss)
print('Accuracy = ',accuracy)

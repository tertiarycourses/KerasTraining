

# Parameters
n_classes = 10
learning_rate = 1
epochs = 2
batch_size = 100

from keras.models import Sequential
from keras.layers import Dense, Dropout,Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# Step 1: Pre-process the data
from keras.datasets import cifar10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Step 2: CNN Model

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu'))
# model.add(MaxPooling2D(2,2))
model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(10,activation="softmax"))

# Step 3: Compile Model

model.compile(optimizer="adam",loss="binary_crossentropy",metrics=['accuracy'])

# Step 4: Train Model

model.fit(X_train,y_train,epochs=2,batch_size=128)

# Step 5: Evaluate Model

score = model.predict(X_test,y_test)
print("loss = ", score[0],"accuracy = ", score[1])

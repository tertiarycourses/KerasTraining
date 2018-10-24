import keras
from keras.models import Sequential
from keras.layers import *
import tensorflow as tf

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Step 1: Preprocess the data

## Preprocess the training data
training_data_df = pd.read_csv('../data/sales_data_training.csv')
scaler = MinMaxScaler(feature_range=(0,1))
scaled_training= scaler.fit_transform(training_data_df)
scaled_training_df = pd.DataFrame(scaled_training,columns= training_data_df.columns.values)
scaled_training_df.to_csv("../data/sales_training_scaled.csv", index=False)

## Preprocess the testing data
testing_data_df = pd.read_csv('../data/sales_data_testing.csv')
scaled_testing= scaler.fit_transform(testing_data_df)
scaled_testing_df = pd.DataFrame(scaled_testing,columns= testing_data_df.columns.values)
scaled_testing_df.to_csv("../data/sales_testing_scaled.csv", index=False)

## Print out the scaling factors
print("Scaled by  {:.10f} and added {:.6f}".format(scaler.scale_[8], scaler.min_[8]))

## Create Input and Outputs for Model training
X = scaled_training_df.drop('total_earnings',axis=1).values
y = scaled_training_df[['total_earnings']].values

## Create Input and Outputs for Model testing
X_test = scaled_testing_df.drop('total_earnings',axis=1).values
y_test = scaled_testing_df[['total_earnings']].values


# Step 2: Build the NN Model
model = Sequential()
model.add(Dense(50,input_dim=9,activation='relu'))
model.add(Dense(100,activation='relu'))
model.add(Dense(200,activation='relu'))
model.add(Dense(1,activation='linear'))
print(model.summary())

# Step 3: Compile the Model
model.compile(loss='mean_squared_error',optimizer='adam')

# Step 4: Train the Model
model.fit(X,y,epochs=50,shuffle=True)

# # Step 5: Evaluate the Model
error = model.evaluate(X_test,y_test)
print("Error = ", error)

model_builder = tf.saved_model.builder.SavedModelBuilder("exported_model4")

inputs = {
    'input': tf.saved_model.utils.build_tensor_info(model.input)
}
outputs = {
    'earnings': tf.saved_model.utils.build_tensor_info(model.output)
}

signature_def = tf.saved_model.signature_def_utils.build_signature_def(
    inputs=inputs,
    outputs=outputs,
    method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME
)

model_builder.add_meta_graph_and_variables(
    K.get_session(),
    tags=[tf.saved_model.tag_constants.SERVING],
    signature_def_map={
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: signature_def
    }
)

model_builder.save()

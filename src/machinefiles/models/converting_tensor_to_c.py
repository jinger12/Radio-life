import tensorflow as tf
from RandomForestRegressor_tensor import model
import os


converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Define the path to your directory
directory_path = '/models'

# Use os.path.join to create the full path to the file
full_file_path = os.path.join(directory_path, 'model.tflite')

# Write the model to the file
with open(full_file_path, 'wb') as f:
    f.write(tflite_model)


##run on command line:
#   xxd -i model.tflite > model.h    ###

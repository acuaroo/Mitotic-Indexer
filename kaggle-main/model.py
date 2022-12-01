from datagen import generate_final_data
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
# (train_images, train_labels), (test_images, test_labels) = generate_final_data()
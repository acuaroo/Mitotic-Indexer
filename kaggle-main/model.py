from datagen import generate_final_data
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
import numpy as np

#if you're interested in using your GPU for this...
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

(train_images, train_labels), (test_images, test_labels), (limit, train_percent) = generate_final_data()

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(40, 40, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

rounded_u = round((limit*2) * train_percent)
round_d = (limit*2) - rounded_u

train_images = train_images.reshape((rounded_u, 40, 40, 1))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((round_d, 40, 40, 1))
test_images = test_images.astype("float32") / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# messing with model
# shuffler = np.random.permutation(len(train_labels))
# train_labels = train_labels[shuffler]

model.fit(train_images, train_labels, epochs=7, batch_size=30)
model.summary()

# messing with model
# shuffler = np.random.permutation(len(test_labels))
# test_labels = test_labels[shuffler]

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)
print(test_loss)
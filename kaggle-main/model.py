from datagen import generate_final_data
import os
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import models
from keras.utils import to_categorical
from tensorflow.keras.utils import array_to_img
import numpy as np
from PIL import Image, ImageOps
from PIL import ImageDraw
from PIL import ImageFont

#if you're interested in using your GPU for this...
#print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

(train_images, train_labels), (test_images, test_labels), (limit, train_percent) = generate_final_data()

model = keras.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(40, 40, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(2, activation="softmax"))

rounded_u = round((limit*2) * train_percent)
round_d = (limit*2) - rounded_u

train_images = train_images.reshape((rounded_u, 40, 40, 3))
train_images = train_images.astype("float32") / 255

test_images = test_images.reshape((round_d, 40, 40, 3))
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

predict_labels = model.predict(test_images)

test_loss, test_acc = model.evaluate(test_images, test_labels)

print(test_acc)
print(test_loss)

# print(test_images[1:2])
# print(np.round(predict_labels[1:10]))
# print(test_labels[1:10])
def concat_images(img_paths, sz, shape=None):
    width, height = sz
    images = map(Image.open, img_paths)
    images = [ImageOps.fit(im, sz, Image.Resampling.LANCZOS)
              for im in images]

    shape = shape if shape else (1, len(images))
    image_size = (width * shape[1], height * shape[0])
    new_image = Image.new('RGB', image_size)

    for row in range(shape[0]):
        for col in range(shape[1]):
            offset = width * col, height * row
            idx = row * shape[1] + col
            new_image.paste(images[idx], offset)

    return new_image

iteration = 0

new_font = ImageFont.truetype('Roboto-Black.ttf', 10)


for cell in test_images[0:100]:
    new_img = array_to_img(test_images[iteration])

    new_img.save('temp-images-clean/'+str(iteration)+'_c.jpg')

    draw_cell = ImageDraw.Draw(new_img)

    draw_cell.text((2, 2), "p"+str(np.round(predict_labels[iteration])), font=new_font, fill=(0, 255, 0))
    draw_cell.text((2, 25), "a"+str(test_labels[iteration]), font=new_font, fill=(0, 255, 0))

    new_img.save('temp-images/'+str(iteration)+'_c.jpg')

    iteration += 1

image_paths = [os.path.join('temp-images', f)
               for f in os.listdir('temp-images') if f.endswith(".jpg")]

concat_final = concat_images(image_paths, (400, 400), (9, 9))
concat_final.show()
concat_final.save('final.jpg')
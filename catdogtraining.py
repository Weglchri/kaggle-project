import os, cv2, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import keras
import tensorflow as tf

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras import layers, models
from keras import backend as K
from sklearn.model_selection import train_test_split
from keras.preprocessing import image


IMG_SIZE = 150

TRAINING_DIR = './catdogfiles/train/'
TESTING_DIR = './catdogfiles/test/'
TEST_DIR = './testimage/'

train_images_dogs_cats = [TRAINING_DIR+i for i in os.listdir(TRAINING_DIR)]
test_images_dogs_cats = [TESTING_DIR+i for i in os.listdir(TESTING_DIR)]

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [ atoi(c) for c in re.split('(\d+)', text) ]

test_images_dogs_cats.sort(key=natural_keys)


def prepare_data(list_of_images):
    """
    Returns two arrays:
        x is an array of resized images
        y is an array of labels
    """
    x = []  # images as arrays
    y = []  # labels


    for image in list_of_images:
        x.append(cv2.resize(cv2.imread(image), (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_CUBIC))

    for i in list_of_images:
        if 'dog' in i:
            y.append(1)
        elif 'cat' in i:
            y.append(0)
    return x, y


X, Y = prepare_data(train_images_dogs_cats)
print(K.image_data_format())

X_train, X_val, Y_train, Y_val = train_test_split(X,Y, test_size=0.2, random_state=1)

nb_train_samples = len(X_train)
nb_validation_samples = len(X_val)
batch_size = 16

# create model
model = models.Sequential()

model.add(layers.Conv2D(32, (3, 3), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(32, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Conv2D(64, (3, 3)))
model.add(layers.Activation('relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

model.add(layers.Flatten())
model.add(layers.Dense(64))
model.add(layers.Activation('relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1))
model.add(layers.Activation('sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#model.summary()

# data generators
training_data_generator = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)
test_data_generator = ImageDataGenerator(rescale=1./255)
validation_data_generator = ImageDataGenerator(rescale=1./255)


# generators
training_generator = training_data_generator.flow(np.array(X_train), Y_train, batch_size=batch_size)
test_generator = test_data_generator.flow_from_directory(
    TEST_DIR,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=100,
    class_mode="binary",
    shuffle=False)
validation_generator = validation_data_generator.flow(np.array(X_val), Y_val, batch_size=batch_size)

# fit the model
history = model.fit_generator (
    training_generator,
    steps_per_epoch=1000,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=500
)

# model.save('models/catdog.h5')
model = load_model('models/catdog.h5')


history.history['acc']
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'validation accuracy'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper left')
plt.show()




img = image.load_img(TEST_DIR + "image.jpg")
img_1 = image.img_to_array(img)
img_1 = cv2.resize(img_1, (IMG_SIZE, IMG_SIZE), interpolation = cv2.INTER_AREA)
img_1 = np.expand_dims(img_1, axis=0) / 255.

# prediction of model
prediction = model.predict(img_1)

plt.imshow(img_1)
plt.axis('off')


if prediction[0] > 0.5:
    plt.title("%.2f" % (prediction[0] * 100) + "% dog")
else:
    plt.title("%.2f" % ((1 - prediction[0]) * 100) + "% cat")
plt.show()


fig, ax = plt.subplots()
ax.imshow(img)
ax.axis('off')
plt.show()

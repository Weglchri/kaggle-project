import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import numpy as np
import os
import shutil
from sklearn.model_selection import train_test_split

import keras
import tensorflow as tf

from keras.models import Model
from keras.layers import Dense
from keras.layers import GlobalAveragePooling2D
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.inception_v3 import InceptionV3


IMG_WIDTH = 150
IMG_HEIGHT = 150

WORKING_PATH = './dog-breed-identification/'

BATCH_SIZE = 16
EPOCHS = 10
TESTING_SPLIT = 0.3	# 70/30 %

NUM_CLASSES = 120
IMAGE_SIZE = 150

labels = pd.read_csv(WORKING_PATH + 'labels.csv')
print(labels.head())

train_ids, valid_ids = train_test_split(labels, test_size = TESTING_SPLIT)

print(len(train_ids), 'train ids', len(valid_ids), 'validation ids')
print('Total', len(labels), 'testing images')


def copyFileSet(strDirFrom, strDirTo, arrFileNames):
    arrBreeds = np.asarray(arrFileNames['breed'])
    arrFileNames = np.asarray(arrFileNames['id'])

    if not os.path.exists(strDirTo):
        os.makedirs(strDirTo)

    for i in tqdm(range(len(arrFileNames))):
        strFileNameFrom = strDirFrom + arrFileNames[i] + ".jpg"
        strFileNameTo = strDirTo + arrBreeds[i] + "/" + arrFileNames[i] + ".jpg"

        if not os.path.exists(strDirTo + arrBreeds[i] + "/"):
            os.makedirs(strDirTo + arrBreeds[i] + "/")

            # As a new breed dir is created, copy 1st file
            # to "test" under name of that breed
            if not os.path.exists(WORKING_PATH + "test/"):
                os.makedirs(WORKING_PATH + "test/")

            strFileNameTo = WORKING_PATH + "test/" + arrBreeds[i] + ".jpg"
            shutil.copy(strFileNameFrom, strFileNameTo)

        shutil.copy(strFileNameFrom, strFileNameTo)


# copyFileSet(WORKING_PATH + "all_images/", WORKING_PATH + "train/", train_ids)
# copyFileSet(WORKING_PATH + "all_images/", WORKING_PATH + "valid/", valid_ids)

breeds = np.unique(labels['breed'])
map_characters = {}
for i in range(len(breeds)):
  map_characters[i] = breeds[i]
  print("<item>" + breeds[i] + "</item>")


def preprocess(img):
    img = cv2.resize(img,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation = cv2.INTER_AREA)
    img_1 = image.img_to_array(img)
    img_1 = cv2.resize(img_1, (IMAGE_SIZE, IMAGE_SIZE),
        interpolation = cv2.INTER_AREA)
    img_1 = np.expand_dims(img_1, axis=0) / 255.
    return img_1[0]


training_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    vertical_flip=False,
    zoom_range=0.3)

validation_data_generator = ImageDataGenerator(
    preprocessing_function=preprocess)

training_generator = training_data_generator.flow_from_directory(
    WORKING_PATH + "train/",
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True,
    class_mode="categorical")

validation_generator = validation_data_generator.flow_from_directory(
    WORKING_PATH + "valid/",
    batch_size=BATCH_SIZE,
    target_size=(IMAGE_SIZE, IMAGE_SIZE),
    shuffle=True,
    class_mode="categorical")


STEP_SIZE_TRAIN=training_generator.n//training_generator.batch_size
STEP_SIZE_VALID=validation_generator.n//validation_generator.batch_size

# import model
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)
for layer in base_model.layers:
    layer.trainable = False
model.compile(optimizer='sgd', loss='categorical_crossentropy',  metrics=['accuracy'])

#model.summary()

history = model.fit_generator(generator=training_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=validation_generator,
        validation_steps=STEP_SIZE_VALID,
        epochs=EPOCHS)


# model.save("models/dogbreed.h5")

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['training accuracy', 'testing accuracy'], loc='upper left')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['training loss', 'validation loss'], loc='upper left')
plt.show()


img = image.load_img(TEST_DIR + "image.jpg")

img = image.load_img(WORKING_PATH + "testimage/image.jpg")
img_1 = image.img_to_array(img)
img_1 = cv2.resize(img_1, (IMAGE_SIZE, IMAGE_SIZE), interpolation = cv2.INTER_AREA)
img_1 = np.expand_dims(img_1, axis=0) / 255.

y_pred = model.predict(img_1)
y_pred_classes = np.argmax(y_pred, axis = 1)

fig, ax = plt.subplots()
ax.imshow(img)
ax.axis('off')
ax.set_title(map_characters[y_pred_classes[0]])
plt.show()







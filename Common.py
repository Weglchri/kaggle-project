import keras
import numpy
import matplotlib.pyplot as plt
from kafka import KafkaConsumer
import pandas
import cv2
from PIL import Image
from keras.models import load_model
from kafka import KafkaConsumer
from keras.preprocessing import image


# To consume latest messages and auto-commit offsets
consumer = KafkaConsumer('dlc1', bootstrap_servers=['localhost:9092'])


def loadImage(path):
    return(image.load_img(path))

def prepareImage(img):
    img_1 = image.img_to_array(img)
    img_1 = cv2.resize(img_1, (150, 150), interpolation=cv2.INTER_AREA)
    img_1 = numpy.expand_dims(img_1, axis=0) / 255.
    return (img_1)


def getDogLabels():
    labels = pandas.read_csv('dog-breed-identification/labels.csv')
    breeds = numpy.unique(labels['breed'])
    map_characters = {}
    for i in range(len(breeds)):
        map_characters[i] = breeds[i]
        print("<item>" + breeds[i] + "</item>")
    return(map_characters)


def getDogBreed(img):
    dogbreedmodel = load_model('models/dogbreed.hdf')
    prediction = dogbreedmodel.predict(img)
    predictedClasses = numpy.argmax(prediction, axis = 1)
    # x = getDogLabels()[predictedClasses[0]]

    fig, ax = plt.subplots()
    ax.imshow(img1)
    ax.axis('off')
    labels = getDogLabels()
    ax.set_title(labels[predictedClasses[0]])
    plt.show()


for img in consumer:
    print(img)
    #receivedImg = numpy.frombuffer(img.value, dtype='float32')
    #prepImg = receivedImg.reshape(1, 150, 150, 3)
    img1 = loadImage(img.value.decode())
    prepImg = prepareImage(img1)
    getDogBreed(prepImg)





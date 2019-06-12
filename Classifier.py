
import keras
import cv2
import numpy
import matplotlib.pyplot as plt
from keras.models import load_model
from kafka import KafkaProducer
from keras.preprocessing import image

producer = KafkaProducer(bootstrap_servers=['localhost:9092'])
catdogmodel = load_model('models/catdog.h5')

def loadImage(path):
    return(image.load_img(path))

def prepareImage(img):
    img_1 = image.img_to_array(img)
    img_1 = cv2.resize(img_1, (150, 150), interpolation=cv2.INTER_AREA)
    img_1 = numpy.expand_dims(img_1, axis=0) / 255.
    return (img_1)

img = loadImage('presentation/tester/random.jpg')
prepImg = prepareImage(img)
prediction = catdogmodel.predict(prepImg)

plt.imshow(img)
plt.axis('off')
if prediction[0] > 0.5:
    plt.title("%.2f" % (prediction[0] * 100) + "% dog")
else:
    plt.title("%.2f" % ((1 - prediction[0]) * 100) + "% cat")
plt.show()


#byteImg = numpy.array(prepImg).tobytes()
#producer.send('dlc1', byteImg) if prediction >= 0.5 else producer.send('dlc2', byteImg)

#encImg = str.encode('presentation/tester/random.jpg')
#producer.send('dlc1', encImg) if prediction >= 0.5 else producer.send('dlc2', encImg)
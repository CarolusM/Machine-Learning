# Modify 'test1.jpg' and 'test2.jpg' to the images you want to predict on
import keras
from keras.datasets import mnist
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
#import matplotlib.pylab as plt
import cv2 as cv
from keras.preprocessing import image
import logging
import sys
import os


#FOR THE LOGGING FILE-------------------------------------------------------
save_dir="/home/cmarquez/Desktop/pruebasTensorFlow"
# Path to a logger file
logging_path = os.path.join(save_dir, "info.log")

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s:%(levelname)s: %(message)s')

console_handler = logging.StreamHandler(sys.stdout)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

file_handler = logging.FileHandler(logging_path)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)
#---------------------------------------------------------------------------


batch_size = 128
num_classes = 10
epochs = 10

# dimensions of our images
img_x, img_y = 28, 28

myimage=cv.imread('zero.jpg',0)
cv.imshow('image',myimage)
cv.waitKey(500)

myimage=cv.resize(myimage,(img_x,img_y))
myimage = image.img_to_array(myimage)
myimage=np.expand_dims(myimage, axis=0)
myimage=myimage.astype('float32')
myimage/=255
images = np.vstack([myimage])
print(images.shape)


# load the model we saved
model = load_model('mnist_easy_model.h5')
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])


classes = model.predict_classes(images, batch_size=10)
print('Prediction:',classes)

preds=model.predict(myimage,verbose=1)
print('Output array:',preds)





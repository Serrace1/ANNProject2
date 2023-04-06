import scipy as sp
import keras as keras
import tensorflow as tf
import matplotlib as mp
import numpy as np
import os, sys
from IPython.display import display
from IPython.display import Image as _Imgdis
from PIL import Image
from time import time
from time import sleep

# Following code is derived from Luis Moneda's code. Link below:
#https://www.kaggle.com/code/lgmoneda/from-image-files-to-numpy-arrays/notebook

folder = './images_original'

files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image_utils as iu

train_files = []
y_train = []
i=0
for _file in files:
    train_files.append(_file)
    label_in_file = _file.title()[0]
    y_train.append(int(label_in_file))

#Unmodified and broken below this line

    # Original Dimensions
image_width = 640
image_height = 480
ratio = 4

image_width = int(image_width / ratio)
image_height = int(image_height / ratio)

channels = 3
nb_classes = 1

dataset = np.ndarray(shape=(len(train_files), channels, image_height, image_width), dtype=np.float32)
i = 0
for _file in train_files:
    img = iu.load_img(folder + "/" + _file)  # this is a PIL image
    img.thumbnail((image_width, image_height))
    # Convert to Numpy Array
    x = iu.img_to_array(img)
    x = x.reshape((3, 120, 160))
    # Normalize
    x = (x - 128.0) / 128.0
    dataset[i] = x
    i += 1
    if i % 250 == 0:
        print("%d images to array" % i)
print("All images to array!")
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
from keras.utils import to_categorical
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import image_utils as iu
from sklearn.model_selection import train_test_split
from keras.applications.resnet import preprocess_input
from keras.applications.resnet import ResNet50
from sklearn import metrics

# Create image and label databases
folder = './images_original'

files = [f for f in os.listdir(folder) if os.path.isfile(os.path.join(folder, f))]

train_files = []
y_train = []
i=0
for _file in files:
    train_files.append(_file)
    label_in_file = _file.title()[0]
    y_train.append(int(label_in_file))

# Original Dimensions
image_width = 432
image_height = 288
channels = 3

#Dataset and Labels
dataset = np.ndarray(shape=(len(train_files), image_height, image_width, channels), dtype=np.float32)
y_train2=np.ndarray(shape=(len(train_files)), dtype=np.float32)


# Width change for sliced versions
image_width_split=144

#Sliced dataset version
dataset_split = np.ndarray(shape=(3*(len(train_files)), image_height, image_width_split, channels), dtype=np.float32)
y_train2_split=np.ndarray(shape=(3*(len(train_files))), dtype=np.float32)

i = 0
for _file in train_files:
    img = iu.load_img(folder + "/" + _file)
    x = iu.img_to_array(img)
    dataset[i] = x
    y_train2[i]=y_train[i]
    splits=np.array_split(x, 3, axis=1)
    y=3*i
    dataset_split[y]=splits[0]
    dataset_split[y+1] = splits[1]
    dataset_split[y+2] = splits[2]
    y_train2_split[y]=y_train[i]
    y_train2_split[y+1] = y_train[i]
    y_train2_split[y+2] = y_train[i]
    i += 1

# 80/20/20 Train/Test/Val splits for normal and sliced versions
X_train, X_test, y_train, y_test = train_test_split(dataset, y_train2, test_size=0.2, shuffle=True, stratify=y_train2)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, shuffle=True, stratify=y_train)

X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(dataset_split, y_train2_split, test_size=0.2, shuffle=True, stratify=y_train2_split)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train_split, y_train_split, test_size=0.25, shuffle=True, stratify=y_train_split)

# Hyperparameters
ep = 10    # epochs
bs = 32   # batch size

# input shape for normal dataset
input_shape = (X_train.shape[1], X_train.shape[2], 3)
resIn=input_shape

########################################################
# Classic CNN Model, normal dataset
print("Classic CNN Model")
model = keras.Sequential()
# 1st conv layer
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
# 2nd conv layer
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
# 3rd conv layer
model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.2))
# 4th conv layer
model.add(keras.layers.Conv2D(64, (2, 2), activation='relu'))
model.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Dropout(0.3))
# flatten output and feed it into dense layer
model.add(keras.layers.Flatten())
# output layer
model.add(keras.layers.Dense(10, activation='softmax'))

# compile and fit the CNN model
optimiser = keras.optimizers.Adam(learning_rate=0.0004)
model.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train and Test
model.fit(X_train, y_train, epochs=ep, batch_size=bs, validation_data=(X_val,y_val), verbose=1)
score = model.evaluate(X_test, y_test, verbose=1)

print("Test loss:", score[0])
print("Test accuracy:", score[1])
yhat=model.predict(X_test)
y_test_OH=to_categorical(y_test)
con=metrics.confusion_matrix(y_test_OH.argmax(axis=1), yhat.argmax(axis=1))
print(con)
##########################################
# Split Classic Model
print("Split Classic Model")

# Input shape for sliced models
input_shape = (X_train_split.shape[1], X_train_split.shape[2], 3)
resIn_split=input_shape

model_split = keras.Sequential()
# 1st conv layer
model_split.add(keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=input_shape))
model_split.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model_split.add(keras.layers.BatchNormalization())
model_split.add(keras.layers.Dropout(0.2))
# 2nd conv layer
model_split.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model_split.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model_split.add(keras.layers.BatchNormalization())
model_split.add(keras.layers.Dropout(0.2))
# 3rd conv layer
model_split.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
model_split.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model_split.add(keras.layers.BatchNormalization())
model_split.add(keras.layers.Dropout(0.2))
# 4th conv layer
model_split.add(keras.layers.Conv2D(64, (2, 2), activation='relu'))
model_split.add(keras.layers.MaxPooling2D((2, 2), strides=(2, 2), padding='same'))
model_split.add(keras.layers.BatchNormalization())
model_split.add(keras.layers.Dropout(0.3))
# flatten output and feed it into dense layer
model_split.add(keras.layers.Flatten())
# output layer
model_split.add(keras.layers.Dense(10, activation='softmax'))

# uses same optimizer as Classic CNN
model_split.compile(optimizer=optimiser,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Train and Test
model_split.fit(X_train_split, y_train_split, epochs=ep, batch_size=bs, validation_data=(X_val_split,y_val_split), verbose=1)
score = model_split.evaluate(X_test_split, y_test_split, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
yhat=model_split.predict(X_test_split)
y_test_OH=to_categorical(y_test_split)
con=metrics.confusion_matrix(y_test_OH.argmax(axis=1), yhat.argmax(axis=1))
print(con)
################################
# ResNet setup

base_model = ResNet50(include_top=False, weights='imagenet')
base_model.trainable = False
base_model_split = ResNet50(include_top=False, weights='imagenet')
base_model_split.trainable = False
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = tf.keras.layers.Dense(10)
print("here")
print(len(base_model.layers))
# ResNet with normal database
print("ResNet Model")
inputs = tf.keras.Input(shape=resIn)
x = preprocess_input(inputs)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model_res = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0004
model_res.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

#Train and Test
history= model_res.fit(X_train, y_train, epochs=ep, batch_size=bs, validation_data=(X_val,y_val), verbose=1)
score = model_res.evaluate(X_test, y_test, verbose=1)
print("Test loss v1:", score[0])
print("Test accuracy v1:", score[1])

# Fine tuning
base_model.trainable = True
for layer in base_model.layers[:40]:
  layer.trainable = False

# Recompile
model_res.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train for another round of epochs, then test again
total_epochs = 2*ep
model_res.fit(X_train, y_train, epochs=total_epochs, initial_epoch=history.epoch[-1], batch_size=bs, validation_data=(X_val,y_val), verbose=1)

score = model_res.evaluate(X_test, y_test, verbose=1)
print("Test loss final:", score[0])
print("Test accuracy final:", score[1])

yhat=model_res.predict(X_test)
y_test_OH=to_categorical(y_test)
con=metrics.confusion_matrix(y_test_OH.argmax(axis=1), yhat.argmax(axis=1))
print(con)
###############################################

# ResNet with sliced database
print("Split ResNet Model")

inputs = tf.keras.Input(shape=resIn_split)
x = preprocess_input(inputs)
x = base_model_split(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model_res_split = tf.keras.Model(inputs, outputs)

base_learning_rate = 0.0004
model_res_split.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train and Test
history=model_res_split.fit(X_train_split, y_train_split, epochs=ep, batch_size=bs, validation_data=(X_val_split,y_val_split), verbose=1)
score = model_res_split.evaluate(X_test_split, y_test_split, verbose=1)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

# Fine tuning
base_model_split.trainable = True
for layer in base_model_split.layers[:100]:
  layer.trainable = False

# Recompile
model_res_split.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=base_learning_rate/10),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

# Train for another round of epochs, then test again
total_epochs = 2*ep
model_res_split.fit(X_train_split, y_train_split, epochs=total_epochs, initial_epoch=history.epoch[-1], batch_size=bs, validation_data=(X_val_split,y_val_split), verbose=1)

score = model_res_split.evaluate(X_test_split, y_test_split, verbose=1)
print("Test loss final:", score[0])
print("Test accuracy final:", score[1])

yhat=model_res_split.predict(X_test_split)
y_test_OH=to_categorical(y_test_split)
con=metrics.confusion_matrix(y_test_OH.argmax(axis=1), yhat.argmax(axis=1))
print(con)
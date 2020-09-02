# Image Classifier using Classic CNN Architecture
# by Brianne Blanchard, 2020

import os
import numpy as np
import tensorflow as tensorflow
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import optimizers

#image width and height
img_width, img_height = 400, 400

# initializing variables for the training and validation data
train_data_dir = r'data/train'
validation_data_dir = r'data/validation'

#rescale the pixel values from [0, 255] to [0, 1] interval
datagen = ImageDataGenerator(rescale = 1./255)

# Collecting Data
# retrieve images and their classes for the train and validation sets
train_generator = datagen.flow_from_directory(
    train_data_dir,
    target_size = (img_width, img_height),
    batch_size = 16,
    class_mode = 'categorical'
)

validation_generator = datagen.flow_from_directory(
    validation_data_dir,
    target_size = (img_width, img_height),
    batch_size = 32,
    class_mode = 'categorical'
)

# Model Architecture Definition
# input -> convolution -> ReLu -> conv. -> ReLu -> pool -> relu -> conv. -> relu -> pool -> fully connected
# The input is a 32 by 32 by 3 array of pixel values
# The first layer is always a convolutional layer
# The activation layer (ReLu) increases the nonlinear properties
# The pool layer reduces the dimensionality of each feature map but retains the most important information
#   It also produces the computational complexity of the network
#   The max takes the largest element from the rectified feature map within a window defined
#       and slides this window over each region the the feature map, taking the max values.
model = Sequential()
model.add(Convolution2D(32, (3,3), input_shape=(img_width, img_height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# Dropout to prevent overfitting
model.add(Flatten())    # to prepare the data for dropout
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(4))
model.add(Activation('softmax'))    # convert the data probabilities to each class  w33

# configure the learning process with compile
#   rmsprop: performs gradient descrnt
#   metrics is accuracy because this is a classification problem
model.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy'])

# augmentation for training
batch_size = 16
nb_epoch = 10
nb_train_samples = 900 * 4
nb_validation_samples = 100 * 4

model.fit(
    train_generator,
    steps_per_epoch = nb_train_samples,
    epochs = nb_epoch,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples
    )

model.save_weights('models/Cat_CNN.h5')

# evaluation
model.evaluate_generator(validation_generator, nb_validation_samples)

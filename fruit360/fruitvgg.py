# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 19:25:24 2019

@author: rahul
"""

from keras.layers import Input, Lambda, Dense, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
import shutil
#from sklearn.metrics import confusion_matrix
import numpy as np
#import matplotlib.pyplot as plt

from glob import glob
import os

def link(src, dst, symlinks=False, ignore=None):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            link(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)
classes=['Apple Braeburn',
         'Apple Golden 1',
         'Apricot',
         'Avocado',
         'banana',
         'Cactus fruit',
         'Cherry 1',
         'Grape Blue',
         'Guava',
         'Kiwi',
         'Lemon',
         'Mango',
         'Mango Red',
         'Onion Red',
         'Orange',
         'Papaya',
         'Peach',
         'Pineapple',
         'Pomegranate',
         'Strawberry'
         ]
os.makedirs("/fruit20/Training")
os.makedirs("/fruit20/Test")

train_path_to="/fruit20/Training"
valid_path_to="/fruit20/Test"
train_path_from="Training"
valid_path_from="Test"

for c in classes:
  link(train_path_from + '/' + c, train_path_to + '/' + c)
  link(valid_path_from + '/' + c, valid_path_to + '/' + c)
  

IMAGE_SIZE = [100, 100] 


epochs = 5
batch_size = 32
train_path = '/fruit20/Training'
valid_path = '/fruit20/Test'

image_files = glob(train_path + '/*/*.jp*g')
valid_image_files = glob(valid_path + '/*/*.jp*g')


vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

for layer in vgg.layers:
  layer.trainable = False

x = Flatten()(vgg.output)
prediction = Dense(20, activation='softmax')(x)


model = Model(inputs=vgg.input, outputs=prediction)


model.summary()


model.compile(
  loss='categorical_crossentropy',
  optimizer='rmsprop',
  metrics=['accuracy']
)



gen = ImageDataGenerator(
  rotation_range=20,
  width_shift_range=0.1,
  height_shift_range=0.1,
  shear_range=0.1,
  zoom_range=0.2,
  horizontal_flip=True,
  vertical_flip=True,
  preprocessing_function=preprocess_input
)





train_generator = gen.flow_from_directory(
  train_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)
valid_generator = gen.flow_from_directory(
  valid_path,
  target_size=IMAGE_SIZE,
  shuffle=True,
  batch_size=batch_size,
)


r = model.fit_generator(
  train_generator,
  validation_data=valid_generator,
  epochs=2,
  steps_per_epoch=len(image_files) // batch_size,
  validation_steps=len(valid_image_files) // batch_size,
)

import tensorflow as tf
model.save("fruit360vgg.h5")
tf.keras.models.save_model(model,"fruit360vgg.h5")

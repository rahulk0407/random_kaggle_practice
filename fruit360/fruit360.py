# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 11:11:57 2019

@author: rahul
"""

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import shutil
test="/test"
train="/training"

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
    

def link(src, dst, symlinks=False, ignore=None):  #for creating less data folder link
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
                
                
# =============================================================================
# def link(src, dst,symlinks=False, ignore=None):
#   if os.path.exists(dst):
#      shutil.copytree(src, dst, symlinks, ignore)
# =============================================================================

train_path_to="/fruit20/Training"
valid_path_to="/fruit20/Test"
train_path_from="Training"
valid_path_from="Test"

for c in classes:
  link(train_path_from + '/' + c, train_path_to + '/' + c)
  link(valid_path_from + '/' + c, valid_path_to + '/' + c)
  
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(20, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255, #image augmentation
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(train_path_to,
                                                    batch_size=40,
                                                    class_mode='categorical',
                                                    target_size=(100, 100))


validation_datagen = ImageDataGenerator(rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

validation_generator = validation_datagen.flow_from_directory(valid_path_to,
                                                              batch_size=40,
                                                              class_mode='categorical',
                                                              target_size=(100, 100))



history = model.fit_generator(train_generator,
                              epochs=10,
                              verbose=2,
                              validation_data=validation_generator)

tf.keras.models.save_model(model,"fruit360.h5")
# =============================================================================
# import keras
# model=keras.models.load_model("fruit360.h5")
# import cv2
# image =cv2.imread("test-multiple_fruits/strawberries4.jpg")
# image=image.reshape(-1,100,100,3)
# pre=model.predict(image)
# 
# =============================================================================

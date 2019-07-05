# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 12:42:39 2019

@author: rahul
"""
import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def get_data(filename):
    with open(filename) as training_file:
        csv_reader = csv.reader(training_file, delimiter=',')
        first_line = True
        temp_images = []
        temp_labels = []
        for row in csv_reader:
            if first_line:
                first_line = False
            else:
                temp_labels.append(row[0])
                image_data = row[1:785]
                image_data_as_array = np.array_split(image_data, 28)
                temp_images.append(image_data_as_array)
        images = np.array(temp_images).astype('float')
        labels = np.array(temp_labels).astype('float')
    return images,labels


training_images, training_labels = get_data('sign_mnist_train.csv')
testing_images, testing_labels = get_data('sign_mnist_test.csv')

print(training_images.shape)
print(training_labels.shape)
print(testing_images.shape)
print(testing_labels.shape)


training_images = np.expand_dims(training_images, axis=3)
testing_images = np.expand_dims(testing_images, axis=3)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

validation_datagen = ImageDataGenerator(
    rescale=1. / 255)

print(training_images.shape)
print(testing_images.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')])

model.compile(optimizer = 'adam',
              loss = 'sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit_generator(train_datagen.flow(training_images, training_labels, batch_size=16),
                              steps_per_epoch=len(training_images) / 16,
                              epochs=12,
                              validation_data=validation_datagen.flow(testing_images, testing_labels, batch_size=16),
                              validation_steps=len(testing_images) / 16)

model.save("sign_language.h5")

model.evaluate(testing_images, testing_labels)
# =============================================================================
# import tensorflow
# model1=tensorflow.keras.models.load_model("sign_language.h5")
# from tensorflow.keras.preprocessing import image
# img = image.load_img('outfile.jpg', target_size=(28, 28,1))
# img=tensorflow.image.rgb_to_grayscale(img)
# output=[]
# sess = tensorflow.Session()
# with sess.as_default():
#     output_img = img.eval()
# x = image.img_to_array(output_img)
# x = np.expand_dims(x, axis=0)
# model.predict(x)
# =============================================================================
# =============================================================================
# 
# from numpy import genfromtxt
# my_data = genfromtxt('sign_mnist_test.csv', delimiter=',')
# import scipy.misc
# scipy.misc.imsave('outfile.jpg', testing_images[0])
# =============================================================================

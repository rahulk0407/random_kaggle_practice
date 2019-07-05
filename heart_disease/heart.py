# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 14:12:19 2019

@author: rahul
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

dataset=pd.read_csv("heart.csv")
dataset.shape
x=dataset.iloc[:,0:13]
y=dataset.iloc[:,13]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(128, activation='relu',input_dim=13),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1,activation='sigmoid')
])

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size = 10, nb_epoch = 10)
y_pred=model.predict(x_test)
y_pred1=(y_pred>0.5).astype(int)
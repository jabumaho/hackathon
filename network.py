import os
import random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import SGD
from tqdm import tqdm

img_size = 256
batch_size = 32
model_name = "{}.h5".format(input("Model name: "))

train_data = np.load('train_data_{}_contrast.npy'.format(img_size), allow_pickle=True)
test_data = np.load('test_data_{}_contrast.npy'.format(img_size), allow_pickle=True)

model = models.Sequential()
model.add(layers.Conv2D(128, (3, 3), activation='relu', input_shape=(img_size, img_size, 1)))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.Conv2D(128, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Dropout(0.2))
model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(2, activation='softmax'))

opt = SGD(learning_rate=0.1)

model.compile(optimizer=opt,
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

X = (np.array([i[0] for i in train_data]) / 255).reshape((-1, img_size, img_size, 1))
Y = np.array([i[1] for i in train_data])

Xtest = (np.array([i[0] for i in test_data]) / 255).reshape((-1, img_size, img_size, 1))
Ytest = np.array([i[1] for i in test_data])

if os.path.exists(model_name):
    model.load_weights(model_name)
    print("Model loaded")

model.fit(X, Y, epochs=int(input("epochs: ")), validation_data=(Xtest, Ytest))

tf.keras.models.save_model(model, model_name)

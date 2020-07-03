import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.optimizers import SGD
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

img_size = 256
model_name = "{}.h5".format(input("Model name: "))

train_data = np.load('train_data_{}.npy'.format(img_size), allow_pickle=True)

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

if os.path.exists(model_name):
    model.load_weights(model_name)
    print("Model loaded")
else:
    print("Model not found")
    exit()


X = (np.array([i[0] for i in train_data]) / 255).reshape((-1, img_size, img_size, 1))
Y = np.array([i[1] for i in train_data])
Y = np.argmax(Y, axis=1)

Y_pred = model.predict_generator(X)
y_pred = np.argmax(Y_pred, axis=1)

print('\nConfusion Matrix:')
print(confusion_matrix(Y, y_pred))
conf_matrix = confusion_matrix(Y, y_pred)
tn, fp, fn, tp = conf_matrix.ravel()
print('\nAccuracy score:')
print(accuracy_score(Y, y_pred))
print('\nF1-score:')
print(f1_score(Y, y_pred))
print('\nPrecision score:')
print(precision_score(Y, y_pred))
print('\nSpecificity:')
specificity = tn / (tn+fp)
print(specificity)
print('\nRecall score:')
print(recall_score(Y, y_pred))


import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
from tensorflow import keras
import os

def plotImage(img, label):
    plt.matshow(img,label=label,cmap=matplotlib.cm.binary)
    plt.xlabel(label)
    plt.show()


def scaleData(X):
    return X / np.amax(X)

def buildModel(m):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(75, 75)),
        tf.keras.layers.Dense(8*m, activation='relu'),
        tf.keras.layers.Dense(6*m, activation='relu'),
        tf.keras.layers.Dense(4*m, activation='relu'),
        tf.keras.layers.Dense(2*m, activation='relu'),
        tf.keras.layers.Dense(m, activation='softmax')
    ])
    model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model

def getLabelNames():
    labels = []
    for r,d,f in os.walk('/home/jelmer/Documents/ML/Week3/Fundus-data'):
        for folder in d:
            labels.append(folder)
    return labels

labelNames = getLabelNames()
list.sort(labelNames)
print(labelNames)
data = []
labels= []
path='/home/jelmer/Documents/ML/Week3/Fundus-data/'
for i in range(len(labelNames)):
    folder = labelNames[i]
    for file in os.listdir(path+folder):
        img = matplotlib.image.imread(path+folder+'/'+file)
        data.append(img)
        labels.append(i)
data = np.array(data)
labels = np.array(labels)
train_amount = 600
X_train, X_test, y_train, y_test = data[:train_amount],data[train_amount:],labels[:train_amount],labels[train_amount:]
shuffle_index = np.random.permutation(train_amount)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

X_train = scaleData(X_train)
X_test = scaleData(X_test)
        
model = buildModel(len(labelNames))
model.fit(X_train,y_train,epochs=130)

model.evaluate(x=X_test,y=y_test)
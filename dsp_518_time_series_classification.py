import numpy as np
import cv2
from os.path import join
import datetime
import math
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split
import tensorflow as tf

# These are the files containing the utility data for each model type in classification
fivep = pd.DataFrame(pd.read_csv('.\PreMA_CSVs\All-Model-Data/all_5P.csv'))
threep = pd.DataFrame(pd.read_csv('.\PreMA_CSVs\All-Model-Data/all_3PC.csv'))
fourp = pd.DataFrame(pd.read_csv('.\PreMA_CSVs\All-Model-Data/all_4PE.csv'))

##################################

##################################
# CONFIGURE DATA
####################################

fs = 24 # sampling frequency in Hz
### Define signal data frames for each change point model type...

# 3P models
x_3P = [np.array(threep['usage'][fs*i:fs*(i+1)], dtype = 'float64') for i in range(np.int(threep.shape[0]/fs))]
x_3P = np.array(x_3P).reshape(np.shape(x_3P))

# 4P models
x_4P = [np.array(fourp['usage'][fs*i:fs*(i+1)], dtype = 'float64') for i in range(np.int(fourp.shape[0]/fs))]
x_4P = np.array(x_4P).reshape(np.shape(x_4P))

# 5P models
x_5P = [np.array(fivep['usage'][fs*i:fs*(i+1)], dtype = 'float64') for i in range(np.int(fivep.shape[0]/fs))]
x_5P = np.array(x_5P).reshape(np.shape(x_5P))

############################################################################################

############################################################################################

############################################################################################
##################################
# EXTRACT LABELS FOR CLASSIFICATION
### 3P == 0
### 4P == 1
### 5P == 2
####################################

model_types = pd.DataFrame(["5P", "3PC", "4P"])
model_codes = model_types[0].astype('category').cat.codes
models_ = model_codes.astype('float64')

fvpLabels = np.zeros(shape = (x_5P.shape[0],1), dtype = 'float64')*-1
tpcLabels = np.ones(shape = (x_3P.shape[0],1), dtype = 'float64')
frpLabels = np.ones(shape = (x_4P.shape[0],1), dtype = 'float64')+1

df_ = np.vstack((x_5P, x_3P, x_4P))
labels_ = np.vstack((fvpLabels, tpcLabels, frpLabels))


# For repeatability, random_seed = 42
np.random.seed(42)
tf.set_random_seed(42)

# Split data into training, test, and validation sets
x_train, x_test, y_train, y_test = train_test_split(df_, labels_, test_size = 0.3)

xv, xt = x_train[:int(x_train.shape[0]/2)], x_train[int(x_train.shape[0]/2):]
yv, yt = y_train[:int(y_train.shape[0]/2)], y_train[int(y_train.shape[0]/2):]

lbls, lbls_counts = np.unique(labels_, return_counts = True)
zero_weight = (1/lbls_counts[0])*(len(labels_)/3) # 5P
one_weight = (1/lbls_counts[1])*(len(labels_)/3) # 3P
two_weight = (1/lbls_counts[2])*(len(labels_)/3) # 4P


class_weights = {0: zero_weight, 1: one_weight, 2: two_weight}#3: three_weight, # This data set has unbalanced distribution of classes
#                 4: four_weight, 5: five_weight}

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.InputLayer(input_shape = (24,)))
model.add(tf.keras.layers.Dense(24, activation = 'sigmoid'))
# model.add(tf.keras.layers.Dense(24, activation = 'sigmoid'))
model.add(tf.keras.layers.Dense(3, activation = 'softmax'))

# import pydot
# import graphviz
# keras.utils.plot_model(model)

weights, biases = model.layers[1].get_weights()

model.compile(loss = 'sparse_categorical_crossentropy',optimizer = 'sgd', metrics = ['accuracy'])

model_history = model.fit(x_train, y_train, epochs = 500, validation_data = (xv,yv), class_weight = class_weights)
# model_history.history to access training metrics

model.evaluate(x_test, y_test)

t= model.predict_classes(x_test)
count_true = 0
count_false = 0
for i in range(int(len(t))):
    if t[i] == y_test[i]: count_true = count_true + 1
    else: count_false = count_false + 1
count_true
count_false

plt.figure(), plt.plot(np.arange(500), model_history.history['val_loss'], label = 'validation loss')
plt.plot(np.arange(500), model_history.history['val_acc'], label = 'validation accuracy')
plt.legend()
plt.show()









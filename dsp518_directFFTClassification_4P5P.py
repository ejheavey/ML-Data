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
fourp = pd.DataFrame(pd.read_csv('.\PreMA_CSVs\All-Model-Data/all_4PE.csv'))
fivep = pd.DataFrame(pd.read_csv('.\PreMA_CSVs\All-Model-Data/all_5P.csv'))
five2 = pd.DataFrame(pd.read_csv('./all_5P_r2-filtered.csv'))
threep = pd.DataFrame(pd.read_csv('./PreMA_CSVs/All-Model-Data/all_3PC.csv'))
#########################################################
#########################################################
#########################################################
#########################################################

def direct_dft(seq, fs, n_pts):
  # Check need to pad sequence...
    x_ = np.array(seq).reshape((len(seq),1))
    pad_length = n_pts - len(x_)
    pads_ = np.zeros(shape = (pad_length, 1))
    x_ = np.vstack((x_, pads_))

    N = len(x_)
    f = np.arange(start = 0, stop = fs, step = 1)
    freq_idx = f/(fs)*N
    freq_idx = freq_idx.reshape((1,len(freq_idx)))
    
  # Begin Goertzel algorithm...
    W_N = np.exp(1j*(2*np.pi/N))
    dft = np.zeros(shape = (fs, 1), dtype = 'complex')
    mag_k = np.zeros(shape = dft.shape, dtype = 'float64')
    for k in range(fs):
        for r in range(N):
            dft[k] = dft[k] + np.dot(x_[r], W_N**(-k*(N-r)))
        mag_k[k] = np.sqrt(np.real(dft[k])**2 + np.imag(dft[k])**2)

  # Shape the return data and write it out...
    X_seq = dft.T/N
    mag_k = mag_k.T/N
    return(X_seq, mag_k, freq_idx)

#########################################################
#########################################################

num_months = 24
x_4P = [np.array(fourp.usage[num_months*i:num_months*(i+1)], dtype = 'float64') for i in range(np.int(fourp.shape[0]/num_months))]
x_4P = np.array(x_4P).reshape(np.shape(x_4P))

fvp = [np.array(fivep.usage[num_months*i:num_months*(i+1)], dtype = 'float64') for i in range(np.int(fivep.shape[0]/num_months))]
fvp = np.array(fvp).reshape(np.shape(fvp))
_5P2 = [np.array(five2.usage[num_months*i:num_months*(i+1)], dtype = 'float64') for i in range(np.int(five2.shape[0]/num_months))]
_5P2 = np.array(_5P2).reshape(np.shape(_5P2))

x_5P = np.vstack((fvp, _5P2))
#########################################################
#########################################################
num_pts = 32
f_sample = 24
#########################################################
#########################################################

dft_5P = np.zeros(shape = (x_5P.shape[0], f_sample), dtype = 'complex')
mag_5P = np.zeros(shape = (x_5P.shape[0], f_sample), dtype = 'float64')
freqs_5P = np.copy(mag_5P)
for l in range(x_5P.shape[0]):
    dft_5P[l], mag_5P[l], freqs_5P[l] = direct_dft(x_5P[l], fs = f_sample, n_pts = num_pts)

dft_4P = np.zeros(shape = (x_5P.shape[0], f_sample), dtype = 'complex')
mag_4P = np.zeros(shape = (x_5P.shape[0], f_sample), dtype = 'float64')
freqs_4P = np.copy(mag_4P)
for l in range(x_5P.shape[0]):
    dft_4P[l], mag_4P[l], freqs_4P[l] = direct_dft(x_4P[l], fs = f_sample, n_pts = num_pts)
#########################################################
#########################################################

#########################################################
#########################################################

frpLabels = np.zeros(shape = (x_5P.shape[0],1), dtype = 'float64')
fvpLabels = np.ones(shape = (x_5P.shape[0],1), dtype = 'float64')

df_ = np.vstack((mag_4P, mag_5P))
labels_ = np.vstack((frpLabels, fvpLabels))


# For repeatability, random_seed = 42
np.random.seed(42)
tf.set_random_seed(42)

# Split data into training, test, and validation sets
x_train, x_test, y_train, y_test = train_test_split(df_, labels_, test_size = 0.3)

import xgboost as xgb

train = xgb.DMatrix(x_train, label = y_train)
test = xgb.DMatrix(x_test[:int(x_test.shape[0]/2)], label = y_test[:int(y_test.shape[0]/2)])

x_valid, y_valid = x_test[int(x_test.shape[0]/2):], y_test[int(y_test.shape[0]/2):]
valid_test = xgb.DMatrix(x_valid, label = y_valid)

params_ = {
            'max_depth': 4,
            'eta': 0.3,
            'objective': 'multi:softmax',
            'num_class': 2}
epochs_ = 10

model_ = xgb.train(params_, train, epochs_)

preds_test = model_.predict(test)
preds_valid = model_.predict(valid_test)

print(preds_test)
print(preds_valid)
from sklearn.metrics import accuracy_score

freq_testingAcc = accuracy_score(y_test[:int(y_test.shape[0]/2)], preds_test)
freq_validationAcc = accuracy_score(y_valid, preds_valid)
avg_freq_Acc = (freq_testingAcc + freq_validationAcc)/2
print("For fs = 24, N = 32, test_size = 0.3:")
print("Freq. testing accuracy: ", freq_testingAcc)

print("Freq. validation accuracy: ", freq_validationAcc)

print("Avg. accuracy score: ", avg_freq_Acc)
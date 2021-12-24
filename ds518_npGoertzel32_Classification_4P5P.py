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

def goertzel_dft(seq, fs, n_pts):
  # Check need to pad sequence...
    seq_ = np.array(seq).reshape((len(seq),1))
    pad_length = n_pts - len(seq_)
    seq_ = np.vstack((seq_, np.zeros(shape = (pad_length,1))))

    N = len(seq_)
    f_res = fs/N
    freq_idx = np.arange(start = 0, stop = N, step = 1)*f_res
    freq_idx = freq_idx.reshape((len(freq_idx),1))

    z = np.exp(1j*2*np.pi*freq_idx)
    W_N = np.exp(1j*(2*np.pi/N))

  # Begin Goertzel algorithm...
    Xz = seq_*z**(-1)
    Hz = 1/(1 - W_N**(-1*freq_idx) * z**(-1))

    dft = Xz*Hz
    magft = [np.sqrt(np.real(entry)**2 + np.imag(entry)**2) for entry in dft]
    magft = np.array(magft).reshape(dft.shape)
    return(dft.T, magft.T, freq_idx.T)


#########################################################
#########################################################

num_months = 24

x_3P = [np.array(threep.usage[num_months*i:num_months*(i+1)], dtype = 'float64') for i in range(np.int(threep.shape[0]/num_months))]
x_3P = np.array(x_3P).reshape(np.shape(x_3P))

x_4P = [np.array(fourp.usage[num_months*i:num_months*(i+1)], dtype = 'float64') for i in range(np.int(fourp.shape[0]/num_months))]
x_4P = np.array(x_4P).reshape(np.shape(x_4P))

fvp = [np.array(fivep.usage[num_months*i:num_months*(i+1)], dtype = 'float64') for i in range(np.int(fivep.shape[0]/num_months))]
fvp = np.array(fvp).reshape(np.shape(fvp))
_5P2 = [np.array(five2.usage[num_months*i:num_months*(i+1)], dtype = 'float64') for i in range(np.int(five2.shape[0]/num_months))]
_5P2 = np.array(_5P2).reshape(np.shape(_5P2))

x_5P = np.vstack((fvp, _5P2))

####################################################
#################################################
###############################
################
num_pts = 32
f_sample = 32


####################################################
#################################################

dft_5P = np.zeros(shape = (x_5P.shape[0], num_pts), dtype = 'complex')
mag_5P = np.zeros(shape = (x_5P.shape[0], num_pts), dtype = 'float64')
freqs_5P = np.copy(mag_5P)
for l in range(x_5P.shape[0]):
    dft_5P[l], mag_5P[l], freqs_5P[l] = goertzel_dft(x_5P[l], fs = f_sample, n_pts = num_pts)

dft_4P = np.zeros(shape = (x_5P.shape[0], num_pts), dtype = 'complex')
mag_4P = np.zeros(shape = (x_5P.shape[0], num_pts), dtype = 'float64')
freqs_4P = np.copy(mag_4P)
for l in range(x_5P.shape[0]):
    dft_4P[l], mag_4P[l], freqs_4P[l] = goertzel_dft(x_4P[l], fs = f_sample, n_pts = num_pts)

###############################
# ################
# plt.figure(),
# for m in range(100):
#     plt.stem(freqs_5P[m], mag_5P[m])
# plt.xlabel('Frequency (Hz)'), plt.ylabel('Magnitude')
# plt.title('Goertzel FFT for 5P (fs = 24, N = 32)')
# 
# plt.figure()
# for m in range(100):
#     plt.stem(freqs_4P[m], mag_4P[m])
# plt.xlabel('Frequency (Hz)'), plt.ylabel('Magnitude')
# plt.title('Goertzel FFT for 4P (fs = 24, N = 32)')
# 
# plt.figure()
# for m in range(100):
#     plt.stem(freqs_3P[m], mag_3P[m])
# plt.xlabel('Frequency (Hz)'), plt.ylabel('Magnitude')
# plt.title('Goertzel FFT for 3PC (fs = 24, N = 32)')

####################################################
#################################################
###############################
################

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














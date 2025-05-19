

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_squared_error
import pickle
import json
from sklearn.feature_selection import SelectKBest, chi2, f_classif, f_regression, mutual_info_regression
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from pyts.classification import TSBF
import math
from scipy.io import arff
import datetime
from scipy.signal import butter,filtfilt
import plotly.graph_objects as go
from scipy.spatial import distance
from numpy import sqrt 
from scipy.stats import entropy
import statsmodels.api as sm
from scipy import stats
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn import metrics
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate, KFold
from collections import Counter
from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn import tree

from FeatureExtraction import dff_phone, df_main
from scipy.spatial.distance import euclidean

#%% Importing Dataset
results = r"C:\Projects\Acceleration-based-activity-recognition\Results"

data_columns = ['X', 'Y', 'Z']
acts = ['A', 'B', 'P', 'R']
#%% Timestamp Preprocessing
ts = []
ts_test =[]
df_main["timestamp"] = pd.to_numeric(df_main["timestamp"], errors="coerce")
df_main = df_main[~df_main["timestamp"].isnull()]
for variable in df_main.timestamp:
    variable = variable/1000000
    new_ts = datetime.datetime.utcfromtimestamp(variable).strftime('%Y-%m-%d %H:%M:%S')
    ts.append(new_ts)

df_main["ts"] = ts
df_main.drop(columns=["timestamp", "Sub_ID"], inplace=True)


#%% Standard Scaling

std_scaler = StandardScaler()

df_scaled_X = std_scaler.fit_transform(df_main.X.to_numpy().reshape(-1,1))
df_scaled_Y = std_scaler.fit_transform(df_main.Y.to_numpy().reshape(-1,1))
df_scaled_Z = std_scaler.fit_transform(df_main.Z.to_numpy().reshape(-1,1))

df_main["X"] = df_scaled_X
df_main["Y"] = df_scaled_Y
df_main["Z"] = df_scaled_Z
#%% Act Preprocessing

df_main['Act'].replace(['A', 'B', 'P', 'R'],
                        [0, 1, 2, 3], inplace=True)

#%% Rolloff Implementation
def windows(dataframe, window_size , overlap_percentage):  
    r = np.arange(len(dataframe))   
    s = r[::overlap_percentage]   
    z = list(zip(s, s + window_size))   
    f = '{0[0]}:{0[1]}'.format
    g = lambda t: dataframe.iloc[t[0]:t[1]]   
    return pd.concat(map(g, z))


rolled_df = windows(df_main, 256, 128)
#%% Low Pass Butterworth Filter

fs = 20.0
cutoff = 0.3
nyq = 0.5 * fs
order = 2

data_X = df_main.X
data_Y = df_main.Y
data_Z = df_main.Z

data_X_gravity = df_main.X - 9.8
data_Y_gravity = df_main.Y - 9.8
data_Z_gravity = df_main.Z - 9.8

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

tBodyAcc_X = butter_lowpass_filter(data_X, cutoff, fs, order)
tBodyAcc_Y = butter_lowpass_filter(data_Y, cutoff, fs, order)
tBodyAcc_Z= butter_lowpass_filter(data_Z, cutoff, fs, order)

'''
tgravityAcc_X = butter_lowpass_filter(data_X_gravity, cutoff, fs, order)
tgravityAcc_Y = butter_lowpass_filter(data_Y_gravity, cutoff, fs, order)
tgravityAcc_Z = butter_lowpass_filter(data_Z_gravity, cutoff, fs, order)

plt.plot(tgravityAcc_X)
plt.plot(tgravityAcc_Y)
plt.plot(tgravityAcc_Z)


plt.plot(tBodyAcc_X)
plt.plot(tBodyAcc_Y)
plt.plot(tBodyAcc_Z)
'''

#%% Jerk signals of time --> da/dt

timeline = pd.to_datetime(df_main["ts"])
what = df_main.X
tBodyAccJerk_X = ((df_main.X.shift(-1) - df_main.X)/50).fillna(0)
tBodyAccJerk_Y = ((df_main.Y.shift(-1) - df_main.Y)/50).fillna(0)
tBodyAccJerk_Z = ((df_main.Z.shift(-1) - df_main.Z)/50).fillna(0)


#%% Euclidean Norm

tBodyAccMag = sqrt((df_main.X.astype(float).shift(-1) - df_main.X.astype(float))**2 + (df_main.Y.astype(float).shift(-1) - df_main.Y.astype(float))**2 + (df_main.Z.astype(float).shift(-1) - df_main.Z.astype(float))**2)
tBodyAccJerkMag = sqrt((tBodyAccJerk_X.shift(-1) - tBodyAccJerk_X)**2 + (tBodyAccJerk_Y.shift(-1) - tBodyAccJerk_Y)**2 + (tBodyAccJerk_Z.shift(-1) - tBodyAccJerk_Z)**2)

#%% FFT implementation

fBodyAcc_X = np.fft.fft(tBodyAcc_X)
fBodyAcc_Y = np.fft.fft(tBodyAcc_Y)
fBodyAcc_Z = np.fft.fft(tBodyAcc_Z)

fBodyAccJerk_X = np.fft.fft(tBodyAccJerk_X)
fBodyAccJerk_Y = np.fft.fft(tBodyAccJerk_Y)
fBodyAccJerk_Z = np.fft.fft(tBodyAccJerk_Z)

fBodyAccJerkMag = np.fft.fft(tBodyAccJerkMag)

# Take the absolute value of the complex numbers for magnitude spectrum
fBodyAcc_magnitude_X = np.abs(fBodyAcc_X)
fBodyAcc_magnitude_Y = np.abs(fBodyAcc_Y)
fBodyAcc_magnitude_Z = np.abs(fBodyAcc_Z)

fBodyAccJerk_magnitude_X = np.abs(fBodyAccJerk_X)
fBodyAccJerk_magnitude_Y = np.abs(fBodyAccJerk_Y)
fBodyAccJerk_magnitude_Z = np.abs(fBodyAccJerk_Z)

 

# Create frequency x-axis that will span up to sample_rate
freq_axis = np.linspace(0, fs, len(fBodyAcc_magnitude_X))

'''
plt.plot(freq_axis, fBodyAcc_magnitude_X)
plt.plot(freq_axis, fBodyAcc_magnitude_Y)
plt.plot(freq_axis, fBodyAcc_magnitude_Z)

plt.plot(freq_axis, fBodyAccJerk_magnitude_X)
plt.plot(freq_axis, fBodyAccJerk_magnitude_Y)
plt.plot(freq_axis, fBodyAccJerk_magnitude_Z)

plt.xlabel("Frequency (Hz)")
plt.xlim(0, 100)
plt.show()
'''
#%% Binning

split_tBodyAcc_X = np.array_split(tBodyAcc_X, len(tBodyAcc_X)/256)
split_tBodyAcc_Y = np.array_split(tBodyAcc_Y, len(tBodyAcc_X)/256)
split_tBodyAcc_Z = np.array_split(tBodyAcc_Z, len(tBodyAcc_X)/256)

split_fBodyAcc_X = np.array_split(fBodyAcc_X, len(tBodyAcc_X)/256)
split_fBodyAcc_Y = np.array_split(fBodyAcc_Y, len(tBodyAcc_X)/256)
split_fBodyAcc_Z = np.array_split(fBodyAcc_Z, len(tBodyAcc_X)/256)

split_tBodyAccJerk_X = np.array_split(tBodyAccJerk_X, len(tBodyAcc_X)/256)
split_tBodyAccJerk_Y = np.array_split(tBodyAccJerk_Y, len(tBodyAcc_X)/256)
split_tBodyAccJerk_Z = np.array_split(tBodyAccJerk_Z, len(tBodyAcc_X)/256)

split_fBodyAcc_magnitude_X = np.array_split(fBodyAcc_magnitude_X, len(tBodyAcc_X)/256)
split_fBodyAcc_magnitude_Y = np.array_split(fBodyAcc_magnitude_Y, len(tBodyAcc_X)/256)
split_fBodyAcc_magnitude_Z = np.array_split(fBodyAcc_magnitude_Z, len(tBodyAcc_X)/256)
#%% Estimation 

############################################# TIME STATISTICS #######################################################################

# 1 - tBodyAcc_mean : Mean Value
tBodyAcc_X_mean = np.array([tBodyAcc_X_mean.mean() for tBodyAcc_X_mean in split_tBodyAcc_X])
tBodyAcc_Y_mean = np.array([tBodyAcc_Y_mean.mean() for tBodyAcc_Y_mean in split_tBodyAcc_Y])
tBodyAcc_Z_mean = np.array([tBodyAcc_Z_mean.mean() for tBodyAcc_Z_mean in split_tBodyAcc_Z])

# 2 - tBodyAcc_std : Standard Deviation
tBodyAcc_X_std = np.array([tBodyAcc_X_std.std() for tBodyAcc_X_std in split_tBodyAcc_X])
tBodyAcc_Y_std = np.array([tBodyAcc_Y_std.std() for tBodyAcc_Y_std in split_tBodyAcc_Y])
tBodyAcc_Z_std = np.array([tBodyAcc_Z_std.std() for tBodyAcc_Z_std in split_tBodyAcc_Z])

# 3 - tBodyAcc_mad : Median absolute deviation
tBodyAcc_X_mad = np.array([stats.median_abs_deviation(x) for x in split_tBodyAcc_X])
tBodyAcc_Y_mad = np.array([stats.median_abs_deviation(y) for y in split_tBodyAcc_Y])
tBodyAcc_Z_mad = np.array([stats.median_abs_deviation(z) for z in split_tBodyAcc_Z])

# 4 - tBodyAcc_max : Maximum value
tBodyAcc_X_max = np.array([tBodyAcc_X_max.max() for tBodyAcc_X_max in split_tBodyAcc_X])
tBodyAcc_Y_max = np.array([tBodyAcc_Y_max.max() for tBodyAcc_Y_max in split_tBodyAcc_Y])
tBodyAcc_Z_max = np.array([tBodyAcc_Z_max.max() for tBodyAcc_Z_max in split_tBodyAcc_Z])

# 5 - tBodyAcc_min : Minimum value 
tBodyAcc_X_min = np.array([tBodyAcc_X_min.min() for tBodyAcc_X_min in split_tBodyAcc_X])
tBodyAcc_Y_min = np.array([tBodyAcc_Y_min.min() for tBodyAcc_Y_min in split_tBodyAcc_Y])
tBodyAcc_Z_min = np.array([tBodyAcc_Z_min.min() for tBodyAcc_Z_min in split_tBodyAcc_Z])

# 6 - tBodyAcc_sma : Signal magnitude area
tBodyAcc_sma = np.sqrt((tBodyAcc_X.astype(float))**2 + (tBodyAcc_Y.astype(float))**2 + (tBodyAcc_Z.astype(float))**2).sum()/len(tBodyAcc_X)

# 7 - tBodyAcc_energy : Energy measure. Sum of the squares divided by the number of values
tBodyAcc_X_energy = np.array([np.square(tBodyAcc_X_square).sum()/len(tBodyAcc_X_square) for tBodyAcc_X_square in split_tBodyAcc_X])
tBodyAcc_Y_energy = np.array([np.square(tBodyAcc_Y_square).sum()/len(tBodyAcc_Y_square) for tBodyAcc_Y_square in split_tBodyAcc_Y])
tBodyAcc_Z_energy = np.array([np.square(tBodyAcc_Z_square).sum()/len(tBodyAcc_Z_square) for tBodyAcc_Z_square in split_tBodyAcc_Z])

# 8 - tBodyAcc_iqr : Interquartile range 
tBodyAcc_X_iqr = np.array([np.subtract(*np.percentile(tBodyAcc_X_iqr, [75,25])) for tBodyAcc_X_iqr in split_tBodyAcc_X])
tBodyAcc_Y_iqr = np.array([np.subtract(*np.percentile(tBodyAcc_Y_iqr, [75,25])) for tBodyAcc_Y_iqr in split_tBodyAcc_Y])
tBodyAcc_Z_iqr = np.array([np.subtract(*np.percentile(tBodyAcc_Z_iqr, [75,25])) for tBodyAcc_Z_iqr in split_tBodyAcc_Z])

# 9 - tBodyAcc_entropy : Signal entropy ------------------> Check Values
tBodyAcc_X_entropy = np.array([entropy(x.astype(float)) for x in split_tBodyAcc_X])
tBodyAcc_X_entropy[np.isneginf(tBodyAcc_X_entropy)] = 0
tBodyAcc_X_entropy[tBodyAcc_X_entropy == 0 ] = np.mean(tBodyAcc_X_entropy)

tBodyAcc_Y_entropy = np.array([entropy(y.astype(float)) for y in split_tBodyAcc_Y])
tBodyAcc_Y_entropy[np.isneginf(tBodyAcc_Y_entropy)] = 0
tBodyAcc_Y_entropy[tBodyAcc_Y_entropy == 0 ] = np.mean(tBodyAcc_Y_entropy)

tBodyAcc_Z_entropy = np.array([entropy(z.astype(float)) for z in split_tBodyAcc_Z])
tBodyAcc_Z_entropy[np.isneginf(tBodyAcc_Z_entropy)] = 0
tBodyAcc_Z_entropy[tBodyAcc_Z_entropy == 0 ] = np.mean(tBodyAcc_Z_entropy)

# 10 - arCoeff : Autorregresion coefficients with Burg order equal to 4
tBodyAcc_X_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(x, order=4) for x in split_tBodyAcc_X])
tBodyAcc_Y_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(y, order=4) for y in split_tBodyAcc_Y])
tBodyAcc_Z_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(z, order=4) for z in split_tBodyAcc_Z])

# 11 - tBodyAcc_Crosscorrelation
tBodyAcc_XY_Crosscorrelation = np.correlate(a=tBodyAcc_X, v=tBodyAcc_Y)
tBodyAcc_XZ_Crosscorrelation = np.correlate(a=tBodyAcc_X, v=tBodyAcc_Z)
tBodyAcc_YZ_Crosscorrelation = np.correlate(a=tBodyAcc_Y, v=tBodyAcc_Z)

############################################# FREQUENCY STATISTICS #######################################################################

# 12 - fBodyAcc_mean : Mean Value
fBodyAcc_X_mean = np.array([fBodyAcc_X_mean.mean() for fBodyAcc_X_mean in split_fBodyAcc_X])
fBodyAcc_Y_mean = np.array([fBodyAcc_Y_mean.mean() for fBodyAcc_Y_mean in split_fBodyAcc_Y])
fBodyAcc_Z_mean = np.array([fBodyAcc_Z_mean.mean() for fBodyAcc_Z_mean in split_fBodyAcc_Z])

# 13 - fBodyAcc_std : Standard Deviation
fBodyAcc_X_std = np.array([fBodyAcc_X_std.std() for fBodyAcc_X_std in split_fBodyAcc_X])
fBodyAcc_Y_std = np.array([fBodyAcc_Y_std.std() for fBodyAcc_Y_std in split_fBodyAcc_Y])
fBodyAcc_Z_std = np.array([fBodyAcc_Z_std.std() for fBodyAcc_Z_std in split_fBodyAcc_Z])

# 14 - fBodyAcc_mad : Median absolute deviation
fBodyAcc_X_mad = np.array([stats.median_abs_deviation(x) for x in split_fBodyAcc_X])
fBodyAcc_Y_mad = np.array([stats.median_abs_deviation(y) for y in split_fBodyAcc_Y])
fBodyAcc_Z_mad = np.array([stats.median_abs_deviation(z) for z in split_fBodyAcc_Z])

# 15 - tBodyAcc_max : Maximum value
fBodyAcc_X_max = np.array([fBodyAcc_X_max.max() for fBodyAcc_X_max in split_fBodyAcc_X])
fBodyAcc_Y_max = np.array([fBodyAcc_Y_max.max() for fBodyAcc_Y_max in split_fBodyAcc_Y])
fBodyAcc_Z_max = np.array([fBodyAcc_Z_max.max() for fBodyAcc_Z_max in split_fBodyAcc_Z])

# 16 - tBodyAcc_min : Minimum value 
fBodyAcc_X_min = np.array([fBodyAcc_X_min.min() for fBodyAcc_X_min in split_fBodyAcc_X])
fBodyAcc_Y_min = np.array([fBodyAcc_Y_min.min() for fBodyAcc_Y_min in split_fBodyAcc_Y])
fBodyAcc_Z_min = np.array([fBodyAcc_Z_min.min() for fBodyAcc_Z_min in split_fBodyAcc_Z])

# 17 - tBodyAcc_sma : Signal magnitude area
fBodyAcc_sma = np.sqrt((fBodyAcc_X.astype(float))**2 + (fBodyAcc_Y.astype(float))**2 + (fBodyAcc_Z.astype(float))**2).sum()/len(fBodyAcc_X)

# 18 - tBodyAcc_energy : Energy measure. Sum of the squares divided by the number of values
fBodyAcc_X_energy = np.array([np.square(fBodyAcc_X_square).sum()/len(fBodyAcc_X_square) for fBodyAcc_X_square in split_fBodyAcc_X])
fBodyAcc_Y_energy = np.array([np.square(fBodyAcc_Y_square).sum()/len(fBodyAcc_Y_square) for fBodyAcc_Y_square in split_fBodyAcc_Y])
fBodyAcc_Z_energy = np.array([np.square(fBodyAcc_Z_square).sum()/len(fBodyAcc_Z_square) for fBodyAcc_Z_square in split_fBodyAcc_Z])

# 19 - tBodyAcc_iqr : Interquartile range 
fBodyAcc_X_iqr = np.array([np.subtract(*np.percentile(fBodyAcc_X_iqr, [75,25])) for fBodyAcc_X_iqr in split_fBodyAcc_X])
fBodyAcc_Y_iqr = np.array([np.subtract(*np.percentile(fBodyAcc_Y_iqr, [75,25])) for fBodyAcc_Y_iqr in split_fBodyAcc_Y])
fBodyAcc_Z_iqr = np.array([np.subtract(*np.percentile(fBodyAcc_Z_iqr, [75,25])) for fBodyAcc_Z_iqr in split_fBodyAcc_Z])

'''
# 20 - tBodyAcc_entropy : Signal entropy ------------------> Check Values
fBodyAcc_X_entropy = np.array([entropy(x) for x in split_fBodyAcc_X])
fBodyAcc_Y_entropy = np.array([entropy(y) for y in split_fBodyAcc_Y])
fBodyAcc_Z_entropy = np.array([entropy(z) for z in split_fBodyAcc_Z])
'''
# 21 - arCoeff : Autorregresion coefficients with Burg order equal to 4
fBodyAcc_X_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(x, order=4) for x in split_fBodyAcc_X])
fBodyAcc_Y_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(y, order=4) for y in split_fBodyAcc_Y])
fBodyAcc_Z_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(z, order=4) for z in split_fBodyAcc_Z])

# 22 - tBodyAcc_Crosscorrelation
fBodyAcc_XY_Crosscorrelation = np.correlate(a=fBodyAcc_X, v=fBodyAcc_Y)
fBodyAcc_XZ_Crosscorrelation = np.correlate(a=fBodyAcc_X, v=fBodyAcc_Z)
fBodyAcc_YZ_Crosscorrelation = np.correlate(a=fBodyAcc_Y, v=fBodyAcc_Z)

############################################# TIME JERK STATISTICS #######################################################################

# 1 - tBodyAccJerk_mean : Mean Value
tBodyAccJerk_X_mean = np.array([tBodyAccJerk_X_mean.mean() for tBodyAccJerk_X_mean in split_tBodyAccJerk_X])
tBodyAccJerk_Y_mean = np.array([tBodyAccJerk_Y_mean.mean() for tBodyAccJerk_Y_mean in split_tBodyAccJerk_Y])
tBodyAccJerk_Z_mean = np.array([tBodyAccJerk_Z_mean.mean() for tBodyAccJerk_Z_mean in split_tBodyAccJerk_Z])

# 2 - tBodyAccJerk_std : Standard Deviation
tBodyAccJerk_X_std = np.array([tBodyAccJerk_X_std.std() for tBodyAccJerk_X_std in split_tBodyAccJerk_X])
tBodyAccJerk_Y_std = np.array([tBodyAccJerk_Y_std.std() for tBodyAccJerk_Y_std in split_tBodyAccJerk_Y])
tBodyAccJerk_Z_std = np.array([tBodyAccJerk_Z_std.std() for tBodyAccJerk_Z_std in split_tBodyAccJerk_Z])

# 3 - tBodyAccJerk_mad : Median absolute deviation
tBodyAccJerk_X_mad = np.array([stats.median_abs_deviation(x) for x in split_tBodyAccJerk_X])
tBodyAccJerk_Y_mad = np.array([stats.median_abs_deviation(y) for y in split_tBodyAccJerk_Y])
tBodyAccJerk_Z_mad = np.array([stats.median_abs_deviation(z) for z in split_tBodyAccJerk_Z])

# 4 - tBodyAccJerk_max : Maximum value
tBodyAccJerk_X_max = np.array([tBodyAccJerk_X_max.max() for tBodyAccJerk_X_max in split_tBodyAccJerk_X])
tBodyAccJerk_Y_max = np.array([tBodyAccJerk_Y_max.max() for tBodyAccJerk_Y_max in split_tBodyAccJerk_Y])
tBodyAccJerk_Z_max = np.array([tBodyAccJerk_Z_max.max() for tBodyAccJerk_Z_max in split_tBodyAccJerk_Z])

# 5 - tBodyAccJerk_min : Minimum value 
tBodyAccJerk_X_min = np.array([tBodyAccJerk_X_min.min() for tBodyAccJerk_X_min in split_tBodyAccJerk_X])
tBodyAccJerk_Y_min = np.array([tBodyAccJerk_Y_min.min() for tBodyAccJerk_Y_min in split_tBodyAccJerk_Y])
tBodyAccJerk_Z_min = np.array([tBodyAccJerk_Z_min.min() for tBodyAccJerk_Z_min in split_tBodyAccJerk_Z])

# 6 - tBodyAccJerk_sma : Signal magnitude area
tBodyAccJerk_sma = np.sqrt((tBodyAccJerk_X.astype(float))**2 + (tBodyAccJerk_Y.astype(float))**2 + (tBodyAccJerk_Z.astype(float))**2).sum()/len(tBodyAccJerk_X)

# 7 - tBodyAccJerk_energy : Energy measure. Sum of the squares divided by the number of values
tBodyAccJerk_X_energy = np.array([np.square(tBodyAccJerk_X_square).sum()/len(tBodyAccJerk_X_square) for tBodyAccJerk_X_square in split_tBodyAccJerk_X])
tBodyAccJerk_Y_energy = np.array([np.square(tBodyAccJerk_Y_square).sum()/len(tBodyAccJerk_Y_square) for tBodyAccJerk_Y_square in split_tBodyAccJerk_Y])
tBodyAccJerk_Z_energy = np.array([np.square(tBodyAccJerk_Z_square).sum()/len(tBodyAccJerk_Z_square) for tBodyAccJerk_Z_square in split_tBodyAccJerk_Z])

# 8 -tBodyAccJerk_iqr : Interquartile range 
tBodyAccJerk_X_iqr = np.array([np.subtract(*np.percentile(tBodyAccJerk_X_iqr, [75,25])) for tBodyAccJerk_X_iqr in split_tBodyAccJerk_X])
tBodyAccJerk_Y_iqr = np.array([np.subtract(*np.percentile(tBodyAccJerk_Y_iqr, [75,25])) for tBodyAccJerk_Y_iqr in split_tBodyAccJerk_Y])
tBodyAccJerk_Z_iqr = np.array([np.subtract(*np.percentile(tBodyAccJerk_Z_iqr, [75,25])) for tBodyAccJerk_Z_iqr in split_tBodyAccJerk_Z])

# 9 - tBodyAccJerk_entropy : Signal entropy ------------------> Check Values
tBodyAccJerk_X_entropy = np.array([entropy(x).astype(float) for x in split_tBodyAccJerk_X])
tBodyAccJerk_X_entropy[np.isneginf(tBodyAccJerk_X_entropy)] = 0
tBodyAccJerk_X_entropy[tBodyAccJerk_X_entropy == 0 ] = np.mean(tBodyAccJerk_X_entropy)

tBodyAccJerk_Y_entropy = np.array([entropy(y).astype(float) for y in split_tBodyAccJerk_Y])
tBodyAccJerk_Y_entropy[np.isneginf(tBodyAccJerk_Y_entropy)] = 0
tBodyAccJerk_Y_entropy[tBodyAccJerk_Y_entropy == 0 ] = np.mean(tBodyAccJerk_Y_entropy)

tBodyAccJerk_Z_entropy = np.array([entropy(z).astype(float) for z in split_tBodyAccJerk_Z])
tBodyAccJerk_Z_entropy[np.isneginf(tBodyAccJerk_Z_entropy)] = 0
tBodyAccJerk_Z_entropy[tBodyAccJerk_Z_entropy == 0 ] = np.mean(tBodyAccJerk_Z_entropy)

# 10 - arCoeff : Autorregresion coefficients with Burg order equal to 4
tBodyAccJerk_X_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(x, order=4) for x in split_tBodyAccJerk_X])
tBodyAccJerk_Y_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(y, order=4) for y in split_tBodyAccJerk_Y])
tBodyAccJerk_Z_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(z, order=4) for z in split_tBodyAccJerk_Z])

# 11 - tBodyAccJerk_Crosscorrelation
tBodyAccJerk_XY_Crosscorrelation = np.correlate(a=tBodyAccJerk_X, v=tBodyAccJerk_Y)
tBodyAccJerk_XZ_Crosscorrelation = np.correlate(a=tBodyAccJerk_X, v=tBodyAccJerk_Z)
tBodyAccJerk_YZ_Crosscorrelation = np.correlate(a=tBodyAccJerk_Y, v=tBodyAccJerk_Z)

######################################################### FREQUENCY MAGNITUDE STATISTICS ################################################################################

# 12 - fBodyAcc_mean : Mean Value
fBodyAcc_magnitude_X_mean = np.array([fBodyAcc_magnitude_X.mean() for fBodyAcc_magnitude_X in split_fBodyAcc_magnitude_X])
fBodyAcc_magnitude_Y_mean = np.array([fBodyAcc_magnitude_Y.mean() for fBodyAcc_magnitude_Y in split_fBodyAcc_magnitude_Y])
fBodyAcc_magnitude_Z_mean = np.array([fBodyAcc_magnitude_Z.mean() for fBodyAcc_magnitude_Z in split_fBodyAcc_magnitude_Z])

# 13 - fBodyAcc_std : Standard Deviation
fBodyAcc_magnitude_X_std = np.array([fBodyAcc_magnitude_X_std.std() for fBodyAcc_magnitude_X_std in split_fBodyAcc_magnitude_X])
fBodyAcc_magnitude_Y_std = np.array([fBodyAcc_magnitude_Y_std.std() for fBodyAcc_magnitude_Y_std in split_fBodyAcc_magnitude_Y])
fBodyAcc_magnitude_Z_std = np.array([fBodyAcc_magnitude_Z_std.std() for fBodyAcc_magnitude_Z_std in split_fBodyAcc_magnitude_Z])

# 14 - fBodyAcc_mad : Median absolute deviation
fBodyAcc_magnitude_X_mad = np.array([stats.median_abs_deviation(x) for x in split_fBodyAcc_magnitude_X])
fBodyAcc_magnitude_Y_mad = np.array([stats.median_abs_deviation(y) for y in split_fBodyAcc_magnitude_Y])
fBodyAcc_magnitude_Z_mad = np.array([stats.median_abs_deviation(z) for z in split_fBodyAcc_magnitude_Z])

# 15 - tBodyAcc_max : Maximum value
fBodyAcc_magnitude_X_max = np.array([fBodyAcc_magnitude_X_max.max() for fBodyAcc_magnitude_X_max in split_fBodyAcc_magnitude_X])
fBodyAcc_magnitude_Y_max = np.array([fBodyAcc_magnitude_Y_max.max() for fBodyAcc_magnitude_Y_max in split_fBodyAcc_magnitude_Y])
fBodyAcc_magnitude_Z_max = np.array([fBodyAcc_magnitude_Z_max.max() for fBodyAcc_magnitude_Z_max in split_fBodyAcc_magnitude_Z])

# 16 - tBodyAcc_min : Minimum value 
fBodyAcc_magnitude_X_min = np.array([fBodyAcc_magnitude_X_min.min() for fBodyAcc_magnitude_X_min in split_fBodyAcc_magnitude_X])
fBodyAcc_magnitude_Y_min = np.array([fBodyAcc_magnitude_Y_min.min() for fBodyAcc_magnitude_Y_min in split_fBodyAcc_magnitude_Y])
fBodyAcc_magnitude_Z_min = np.array([fBodyAcc_magnitude_Z_min.min() for fBodyAcc_magnitude_Z_min in split_fBodyAcc_magnitude_Z])

# 17 - tBodyAcc_sma : Signal magnitude area
fBodyAcc_magnitude_sma = np.sqrt((fBodyAcc_magnitude_X.astype(float))**2 + (fBodyAcc_magnitude_Y.astype(float))**2 + (fBodyAcc_magnitude_Z.astype(float))**2).sum()/len(fBodyAcc_magnitude_Z)

# 18 - tBodyAcc_energy : Energy measure. Sum of the squares divided by the number of values
fBodyAcc_magnitude_X_energy = np.array([np.square(fBodyAcc_magnitude_X_square).sum()/len(fBodyAcc_magnitude_X_square) for fBodyAcc_magnitude_X_square in split_fBodyAcc_magnitude_X])
fBodyAcc_magnitude_Y_energy = np.array([np.square(fBodyAcc_magnitude_Y_square).sum()/len(fBodyAcc_magnitude_Y_square) for fBodyAcc_magnitude_Y_square in split_fBodyAcc_magnitude_Y])
fBodyAcc_magnitude_Z_energy = np.array([np.square(fBodyAcc_magnitude_Z_square).sum()/len(fBodyAcc_magnitude_Z_square) for fBodyAcc_magnitude_Z_square in split_fBodyAcc_magnitude_Z])

# 19 - tBodyAcc_iqr : Interquartile range 
fBodyAcc_magnitude_X_iqr = np.array([np.subtract(*np.percentile(fBodyAcc_magnitude_X_iqr, [75,25])) for fBodyAcc_magnitude_X_iqr in split_fBodyAcc_magnitude_X])
fBodyAcc_magnitude_Y_iqr = np.array([np.subtract(*np.percentile(fBodyAcc_magnitude_Y_iqr, [75,25])) for fBodyAcc_magnitude_Y_iqr in split_fBodyAcc_magnitude_Y])
fBodyAcc_magnitude_Z_iqr = np.array([np.subtract(*np.percentile(fBodyAcc_magnitude_Z_iqr, [75,25])) for fBodyAcc_magnitude_Z_iqr in split_fBodyAcc_magnitude_Z])


# 20 - tBodyAcc_entropy : Signal entropy ------------------> Check Values
fBodyAcc_magnitude_X_entropy = np.array([entropy(x) for x in split_fBodyAcc_magnitude_X])
fBodyAcc_magnitude_Y_entropy = np.array([entropy(y) for y in split_fBodyAcc_magnitude_Y])
fBodyAcc_magnitude_Z_entropy = np.array([entropy(z) for z in split_fBodyAcc_magnitude_Z])

# 21 - arCoeff : Autorregresion coefficients with Burg order equal to 4
fBodyAcc_magnitude_X_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(x, order=4) for x in split_fBodyAcc_magnitude_X])
fBodyAcc_magnitude_Y_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(y, order=4) for y in split_fBodyAcc_magnitude_Y])
fBodyAcc_magnitude_Z_arCoeff_arSigma = np.array([sm.regression.linear_model.burg(z, order=4) for z in split_fBodyAcc_magnitude_Z])

# 22 - tBodyAcc_Crosscorrelation
fBodyAcc_XY_Crosscorrelation = np.correlate(a=fBodyAcc_magnitude_X, v=fBodyAcc_magnitude_Y)
fBodyAcc_XZ_Crosscorrelation = np.correlate(a=fBodyAcc_magnitude_X, v=fBodyAcc_magnitude_Z)
fBodyAcc_YZ_Crosscorrelation = np.correlate(a=fBodyAcc_magnitude_Y, v=fBodyAcc_magnitude_Z)
#%% Feature Vector 

x_train = pd.DataFrame()

############################################################# TIME STATISTICS ADDITION #################################################

x_train['tBodyAcc_X_mean'] = tBodyAcc_X_mean.tolist()
x_train['tBodyAcc_Y_mean'] = tBodyAcc_Y_mean.tolist()
x_train['tBodyAcc_Z_mean'] = tBodyAcc_Z_mean.tolist()

x_train['tBodyAcc_X_std'] =  tBodyAcc_X_std.tolist()
x_train['tBodyAcc_Y_std'] =  tBodyAcc_Y_std.tolist()
x_train['tBodyAcc_Z_std'] =  tBodyAcc_Z_std.tolist()

x_train['tBodyAcc_X_mad'] = tBodyAcc_X_mad.tolist()
x_train['tBodyAcc_Y_mad'] = tBodyAcc_Y_mad.tolist()
x_train['tBodyAcc_Z_mad'] = tBodyAcc_Z_mad.tolist()

x_train['tBodyAcc_X_max'] = tBodyAcc_X_max.tolist()
x_train['tBodyAcc_Y_max'] = tBodyAcc_Y_max.tolist()
x_train['tBodyAcc_Z_max'] = tBodyAcc_Z_max.tolist()

x_train['tBodyAccJerk_X_min'] = tBodyAccJerk_X_min.tolist()
x_train['tBodyAccJerk_Y_min'] = tBodyAccJerk_Y_min.tolist()
x_train['tBodyAccJerk_Z_min'] = tBodyAccJerk_Z_min.tolist()

x_train['tBodyAcc_sma'] = tBodyAcc_sma.tolist()

x_train['tBodyAcc_X_energy'] = tBodyAcc_X_energy.tolist()
x_train['tBodyAcc_Y_energy'] = tBodyAcc_Y_energy.tolist()
x_train['tBodyAcc_Z_energy'] = tBodyAcc_Z_energy.tolist()

x_train['tBodyAcc_X_iqr'] = tBodyAcc_X_iqr.tolist()
x_train['tBodyAcc_Y_iqr'] = tBodyAcc_Y_iqr.tolist()
x_train['tBodyAcc_Z_iqr'] = tBodyAcc_Z_iqr.tolist()

x_train['tBodyAcc_X_entropy'] = tBodyAcc_X_entropy.tolist()
x_train['tBodyAcc_Y_entropy'] = tBodyAcc_Y_entropy.tolist()
x_train['tBodyAcc_Z_entropy'] = tBodyAcc_Z_entropy.tolist()


############################################################# TIME JERK STATISTICS ADDITION #################################################

x_train['tBodyAccJerk_X_mean'] = tBodyAccJerk_X_mean.tolist()
x_train['tBodyAccJerk_Y_mean'] = tBodyAccJerk_Y_mean.tolist()
x_train['tBodyAccJerk_Z_mean'] = tBodyAccJerk_Z_mean.tolist()

x_train['tBodyAccJerk_X_std'] =  tBodyAccJerk_X_std.tolist()
x_train['tBodyAccJerk_Y_std'] =  tBodyAccJerk_Y_std.tolist()
x_train['tBodyAccJerk_Z_std'] =  tBodyAccJerk_Z_std.tolist()

x_train['tBodyAccJerk_X_mad'] = tBodyAccJerk_X_mad.tolist()
x_train['tBodyAccJerk_Y_mad'] = tBodyAccJerk_Y_mad.tolist()
x_train['tBodyAccJerk_Z_mad'] = tBodyAccJerk_Z_mad.tolist()

x_train['tBodyAccJerk_X_max'] = tBodyAccJerk_X_max.tolist()
x_train['tBodyAccJerk_Y_max'] = tBodyAccJerk_Y_max.tolist()
x_train['tBodyAccJerk_Z_max'] = tBodyAccJerk_Z_max.tolist()

x_train['tBodyAcc_X_min'] = tBodyAcc_X_min.tolist()
x_train['tBodyAcc_Y_min'] = tBodyAcc_Y_min.tolist()
x_train['tBodyAcc_Z_min'] = tBodyAcc_Z_min.tolist()

x_train['tBodyAccJerk_sma'] = tBodyAccJerk_sma.tolist()

x_train['tBodyAccJerk_X_energy'] = tBodyAccJerk_X_energy.tolist()
x_train['tBodyAccJerk_Y_energy'] = tBodyAccJerk_Y_energy.tolist()
x_train['tBodyAccJerk_Z_energy'] = tBodyAccJerk_Z_energy.tolist()

x_train['tBodyAccJerk_X_iqr'] = tBodyAccJerk_X_iqr.tolist()
x_train['tBodyAccJerk_Y_iqr'] = tBodyAccJerk_Y_iqr.tolist()
x_train['tBodyAccJerk_Z_iqr'] = tBodyAccJerk_Z_iqr.tolist()


############################################################# FREQUENCY MAGNITUDE ADDITION #################################################

x_train['fBodyAcc_magnitude_X_mean'] = fBodyAcc_magnitude_X_mean.tolist()
x_train['fBodyAcc_magnitude_Y_mean'] = fBodyAcc_magnitude_Y_mean.tolist()
x_train['fBodyAcc_magnitude_Z_mean'] = fBodyAcc_magnitude_Z_mean.tolist()

x_train['fBodyAcc_magnitude_X_std'] =  fBodyAcc_magnitude_X_std.tolist()
x_train['fBodyAcc_magnitude_Y_std'] =  fBodyAcc_magnitude_Y_std.tolist()
x_train['fBodyAcc_magnitude_Z_std'] =  fBodyAcc_magnitude_Z_std.tolist()

x_train['fBodyAcc_magnitude_X_mad'] = fBodyAcc_magnitude_X_mad.tolist()
x_train['fBodyAcc_magnitude_Y_mad'] = fBodyAcc_magnitude_Y_mad.tolist()
x_train['fBodyAcc_magnitude_Z_mad'] = fBodyAcc_magnitude_Z_mad.tolist()

x_train['fBodyAcc_magnitude_X_max'] = fBodyAcc_magnitude_X_max.tolist()
x_train['fBodyAcc_magnitude_Y_max'] = fBodyAcc_magnitude_Y_max.tolist()
x_train['fBodyAcc_magnitude_Z_max'] = fBodyAcc_magnitude_Z_max.tolist()

x_train['fBodyAcc_magnitude_X_min'] = fBodyAcc_magnitude_X_min.tolist()
x_train['fBodyAcc_magnitude_Y_min'] = fBodyAcc_magnitude_Y_min.tolist()
x_train['fBodyAcc_magnitude_Z_min'] = fBodyAcc_magnitude_Z_min.tolist()

x_train['fBodyAcc_magnitude_sma'] = fBodyAcc_magnitude_sma.tolist()

x_train['fBodyAcc_magnitude_X_energy'] = fBodyAcc_magnitude_X_energy.tolist()
x_train['fBodyAcc_magnitude_Y_energy'] = fBodyAcc_magnitude_Y_energy.tolist()
x_train['fBodyAcc_magnitude_Z_energy'] = fBodyAcc_magnitude_Z_energy.tolist()

x_train['fBodyAcc_magnitude_X_iqr'] = fBodyAcc_magnitude_X_iqr.tolist()
x_train['fBodyAcc_magnitude_Y_iqr'] = fBodyAcc_magnitude_Y_iqr.tolist()
x_train['fBodyAcc_magnitude_Z_iqr'] = fBodyAcc_magnitude_Z_iqr.tolist()


y_train = np.array_split(df_main.Act, len(tBodyAcc_X)/256)
y_train = [max(var.mode()) for var in y_train]


#%% ML Model
random_state = 0
test_size = 0.3
X_train, X_test, y_train, y_test = train_test_split(x_train , y_train, test_size=test_size, 
                                                    random_state=random_state)
labels=['Walking','Jogging', 'Driblling','Clapping']


from funcs import plot_confusion_matrix, perform_model, print_grid_search_attributes
from sklearn import linear_model
from sklearn import metrics
from sklearn.model_selection import GridSearchCV


# start Grid searc

from sklearn.svm import LinearSVC
parameters = {'C':[0.125, 0.5, 1, 2, 8, 16]}
lr_svc = LinearSVC(tol=0.00005,max_iter=2000)
lr_svc_grid = GridSearchCV(lr_svc, param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
lr_svc_grid_results = perform_model(lr_svc_grid, X_train, y_train, X_test, y_test, class_labels=labels)
# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(lr_svc_grid_results['confusion_matrix'], classes=labels, cmap=plt.cm.Greens, )
plt.show()
print_grid_search_attributes(lr_svc_grid_results['model'])


####knn
from sklearn.neighbors import KNeighborsClassifier
parameters = {'n_neighbors':[3,5,11, 19, 23],
              'weights':['uniform', 'distance'], 
              'metric':['euclidean', 'manhattan']}
knn = KNeighborsClassifier()
knn_grid = GridSearchCV(knn,param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
knn_grid_results = perform_model(knn_grid, X_train, y_train, X_test, y_test, class_labels=labels)
# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(knn_grid_results['confusion_matrix'], classes=labels, cmap=plt.cm.Greens, )
plt.show()
print_grid_search_attributes(knn_grid_results['model'])



####svm with rbf kernel
from sklearn.svm import SVC
parameters = {'C':[2,8,16],\
              'gamma': [ 0.0078125, 0.125, 2]}
rbf_svm = SVC(kernel='rbf')
rbf_svm_grid = GridSearchCV(rbf_svm,param_grid=parameters, cv=3, verbose=1, n_jobs=-1)
rbf_svm_grid_results = perform_model(rbf_svm_grid, X_train, y_train, X_test, y_test, class_labels=labels)
# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(rbf_svm_grid_results['confusion_matrix'], classes=labels, cmap=plt.cm.Greens, )
plt.show()
print_grid_search_attributes(rbf_svm_grid_results['model'])


##### random forest
from sklearn.ensemble import RandomForestClassifier
params = {'n_estimators': np.arange(10,201,20), 'max_depth':np.arange(3,15,2)}
rfc = RandomForestClassifier()
rfc_grid = GridSearchCV(rfc, param_grid=params, cv=3, verbose=1, n_jobs=-1)
rfc_grid_results = perform_model(rfc_grid, X_train, y_train, X_test, y_test, class_labels=labels)
# plot confusion matrix
plt.figure(figsize=(8,8))
plt.grid(b=False)
plot_confusion_matrix(rfc_grid_results['confusion_matrix'], classes=labels, cmap=plt.cm.Greens, )
plt.show()
print_grid_search_attributes(rfc_grid_results['model'])
# mse = mean_squared_error(y_test, y_pred)
# score = clf.score(X_test, y_test)
# classification_report = classification_report(y_test, y_pred)
# feat_importances = pd.Series(clf.feature_importances_, index=x_train.columns)
# feat_importances.nlargest().plot(kind='bar')

# def plot_classification_report(y_tru, y_prd, figsize=(10, 10), ax=None):

#     plt.figure(figsize=figsize)

#     xticks = ['precision', 'recall', 'f1-score', 'support']
#     yticks = list(np.unique(y_tru))
#     yticks += ['avg']

#     rep = np.array(precision_recall_fscore_support(y_test, y_pred)).T
#     avg = np.mean(rep, axis=0)
#     avg[-1] = np.sum(rep[:, -1])
#     rep = np.insert(rep, rep.shape[0], avg, axis=0)

#     sns.heatmap(rep,
#                 annot=True, 
#                 cbar=False, cmap="crest",
#                 xticklabels=xticks, 
#                 yticklabels=yticks,
#                 ax=ax)

# plot_classification_report(y_test, y_pred)

#%% Save model 

# save_stuff = results + "\model" + ".pkl"
# pickle.dump(clf, open(save_stuff, 'wb'))
# '''
# '''

save_path = "C:\Projects\Acceleration-based-activity-recognition\Saved-Models"+ "\model1" + ".sav"
pickle.dump(rfc_grid, open(save_path, 'wb'))

#Testing Model



# f, ax = plt.subplots(figsize=(10, 8))
# corr = df_main.corr()
# sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
#             cmap=sns.diverging_palette(220, 10, as_cmap=True),
#             square=True, ax=ax)
# #%% Time Series Bag-Of-Features

# random_state = 0
# test_size = 0.3
# X_train, X_test, y_train, y_test = train_test_split(x_train, y_train, test_size=test_size, random_state=random_state)

# clf = TSBF(bins=2)
# clf.fit(X_train, y_train)
# y_pred = clf.predict(X_test)
# something= f1_score(y_test, y_pred, average='micro')
# print(something)
# '''
# '''
# plt.bar(np.arange(clf.n_features_in_), clf.feature_importances_)
# plt.title('Feature importance scores')
# plt.xticks(np.arange(clf.n_features_in_),
#             ['feature {}'.format(i) for i in range(clf.n_features_in_)],
#             rotation=90)
# plt.ylabel("Mean decrease in impurity")
# plt.tight_layout()
# plt.show()



# x_data_test = pd.read_excel("C:\Motion Based Acceleration Detection\Excel_Files\Results\ForTestingPurposes.xlsx")
# model_path = r"C:\Motion Based Acceleration Detection\Excel_Files\Results\model.sav"
# rolloff_mean_test = x_data_test[data_columns].rolling(7, center=True).mean()
# rolloff_median_test = x_data_test[data_columns].rolling(7, center=True).median()
# rolloff_std_test = x_data_test[data_columns].rolling(7, center=True).std()

# df_stats_test = result = pd.concat([rolloff_mean_test, rolloff_median_test, rolloff_std_test], axis=1, join="inner").dropna()

# x_data_test = df_stats_test[["X","Y","Z"]]
# with open(model_path, "rb") as f:
#     loaded_model = pickle.load(f)
#     what_is_the_result= loaded_model.predict(df_stats_test)
#     print(what_is_the_result)

# https://builtin.com/data-science/time-series-python
# https://hal.inria.fr/hal-03558165/document ----> "Time series bag-of-features"
# https://towardsdatascience.com/time-series-classification-using-dynamic-time-warping-61dcd9e143f6
# https://notebook.community/alistairwalsh/nefarious-meow/simplify%20clustering_example
# https://www.kaggle.com/code/zadehmoradian/classification-human-activity-recognition-with-sph/notebook
# https://medium.com/@rubeen.786.mr/human-activity-recognition-har-db5c1432cd98
# https://www.c-sharpcorner.com/article/sensor-manager-with-example-in-android/
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Preprocessing import create_dataset
from sklearn.decomposition import PCA
import seaborn as sns

df_phone=[]
df_watch=[]

for i in range(1601,1651):
    df_phone.append(pd.read_csv('Data/' + 'data_' + str(i)+ '_accel_phone.csv')) 
df_main = pd.concat([df_phone[i] for i in range(50)],
                    ignore_index=True, sort=False)
              
dff_phone = pd.read_csv('Data/data_1600_accel_phone.csv')
# dff_watch = pd.read_csv('Data+w/data_1600_accel_watch.csv')
# dff = pd.concat([dff_phone, dff_watch], ignore_index=True, sort=False)

##Feature creating
data_X = df_main.X
data_Y = df_main.Y
data_Z = df_main.Z

split_X = np.array_split(data_X, 75)
split_Y = np.array_split(data_Y, 75)
split_Z = np.array_split(data_Z, 75)

X_mean = np.array([XX_mean.mean() for XX_mean in split_X])
Y_mean = np.array([YY_mean.mean() for YY_mean in split_Y])
Z_mean = np.array([ZZ_mean.mean() for ZZ_mean in split_Z])

X_std = np.array([XX_std.std() for XX_std in split_X])
Y_std = np.array([YY_std.std() for YY_std in split_Y])
Z_std = np.array([ZZ_std.std() for ZZ_std in split_Z])

X_max = np.array([XX_max.max() for XX_max in split_X])
Y_max = np.array([YY_max.max() for YY_max in split_Y])
Z_max = np.array([ZZ_max.max() for ZZ_max in split_Z])

X_min = np.array([XX_min.min() for XX_min in split_X])
Y_min = np.array([YY_min.min() for YY_min in split_Y])
Z_min = np.array([ZZ_min.min() for ZZ_min in split_Z])

X_var = np.array([XX_var.var() for XX_var in split_X])
Y_var = np.array([YY_var.var() for YY_var in split_Y])
Z_var = np.array([ZZ_var.var() for ZZ_var in split_Z])


XY_Crosscorrelation = np.correlate(a=data_X, v=data_Y)
XZ_Crosscorrelation = np.correlate(a=data_Y, v=data_Z)
YZ_Crosscorrelation = np.correlate(a=data_Z, v=data_X)


features = ['X_mean', 'Y_mean', 'Z_mean', 
                'X_std', 'Y_std', 'Z_std', 
                'X_max', 'Y_max', 'Z_max',
                'X_min', 'Y_min', 'Z_min',
                'X_var', 'Y_var', 'Z_var']

vectors = np.array([X_mean, Y_mean , Z_mean, 
                X_std, Y_std, Z_std, 
                X_max, Y_max, Z_max,
                X_min, Y_min, Z_min,
                X_var, Y_var, Z_var])

vectors = np.transpose(vectors)
features_df = pd.DataFrame(data = vectors, columns = features)

df_tot_feat = pd.concat([df_main, features_df], axis = 1)
X_train, y_train, X_test, y_test = create_dataset(df_tot_feat)

# pca = PCA(n_components = 5)
# X_train = pca.fit_transform(X_train)
# X_test = pca.transform(X_test)

# f, ax = plt.subplots(figsize=(10, 8))
# corr = df_tot_feat.corr()
# sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
#             cmap=sns.diverging_palette(220, 10, as_cmap=True),
#             square=True, ax=ax)



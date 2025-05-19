import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('Data/data_1601_accel_phone.csv')
time_stamps = len(df.loc[df['Act'].isin(['A'])].timestamp)

print(df.groupby("Act").X.mean())
print(df.groupby("Act").Y.mean())
print(df.groupby("Act").Z.mean())

print(df.groupby("Act").X.agg(['min', 'max', 'mean', 'median', 'count']))
print(df.groupby("Act").Y.agg(['min', 'max', 'mean', 'median', 'count']))
print(df.groupby("Act").Z.agg(['min', 'max', 'mean', 'median', 'count']))

print(df.X.agg(['min', 'max', 'mean', 'median', 'count']))
print(df.Y.agg(['min', 'max', 'mean', 'median', 'count']))
print(df.Z.agg(['min', 'max', 'mean', 'median', 'count']))


plt.figure(figsize=(12,5))

plt.subplot(3,1,1)
plt.plot(np.arange(0,time_stamps/25, 0.04), df.loc[df['Act'].isin(['A'])].X, label ='X-axis')
plt.title('Walking Activity recorded using phone accelerometer')
plt.legend()

plt.subplot(3,1,2)
plt.plot(np.arange(0,time_stamps/25, 0.04), df.loc[df['Act'].isin(['A'])].Y, label ='Y-axis')
plt.ylabel('Amplitude')
plt.legend()

plt.subplot(3,1,3)
plt.plot(np.arange(0,time_stamps/25, 0.04), df.loc[df['Act'].isin(['A'])].Z, label ='Z-axis')
plt.xlabel('Time (seconds)')
plt.legend()

plt.show()



plt.figure(figsize=(12,5))

# first plot: X-axis against timestamp
plt.subplot(3,1,1)
plt.plot(np.arange(0,time_stamps/25 +0.04, 0.04), df.loc[df['Act'].isin(['B'])].X, label ='X-axis')
plt.title('Jogging Activity recorded using phone accelerometer')
plt.legend()

# add second plot: Y-axis against timestamp, give an appropriate label
plt.subplot(3,1,2)
plt.plot(np.arange(0,time_stamps/25 +0.04, 0.04), df.loc[df['Act'].isin(['B'])].Y, label ='Y-axis')
plt.ylabel('Amplitude')
plt.legend()

# add third plot: Z-axis against timestamp, give an appropriate label
plt.subplot(3,1,3)
plt.plot(np.arange(0,time_stamps/25 +0.04, 0.04), df.loc[df['Act'].isin(['B'])].Z, label ='Z-axis')
plt.xlabel('Time (seconds)')
plt.legend()

plt.show()




























































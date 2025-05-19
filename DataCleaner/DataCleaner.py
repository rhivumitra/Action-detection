
import pandas as pd
import numpy as np

def cleaner(i):
    """
    Convering .txt file to .csv and labeling the columns
    Taking samples with actions: A,B,R,P
    
    """
    
    df0 = pd.read_csv('Data/' + 'data_' + str(i)+ '_accel_phone.txt')
    df0.to_csv('data_' + str(i)+ '_accel_phone.csv', index=None)   
    
    df0.columns = ['Sub_ID', 'Act', 'timestamp', 'X', 'Y', 'Z']
    df0.to_csv('data_' + str(i)+ '_accel_phone.csv', index=None)
    
    df1 = df0.loc[df0['Act'].isin(['A', 'B', 'P', 'R'])]
    df1.to_csv('data_' + str(i)+ '_accel_phone.csv', index=False)
    return df1

##One-time use###
for i in range(1600,1651,1):
    cleaner(i)


def remove_semicolon(j):
    """
    Removing possible semicolons at the end of the last columns values
    Converting their type from strinng to numpy
    
    """
    df = pd.read_csv('data_' + str(j)+ '_accel_phone.csv')
    
    for i in range(len(df.iloc[:,-1])):
        df.iloc[:,-1][i] = df.iloc[:,-1][i].replace(";", " " )
        df.iloc[:,-1][i] = np.array(df.iloc[:,-1][i])
        df.iloc[:,-1][i] = df.iloc[:,-1][i].astype(dtype = np.float64)
        
    df.to_csv('data_' + str(j)+ '_accel_phone.csv', index=False)
    return df

#One-time use###
for i in range(1600,1651,1):
    remove_semicolon(i)































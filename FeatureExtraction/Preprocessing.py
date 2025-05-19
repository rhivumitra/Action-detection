import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
def create_dataset(df):
    ###importing dataset
    # df = pd.read_csv('Data/data_1601_accel_phone.csv')
    # df['time_index'] =np.arange(0,len(df)/25, 0.04)
    
    # df.set_index('time_index')
    X = df.iloc[:, 3:].values
    y = df.iloc[:, 1].values
    
    ###Taking Care of missing data
    
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X)
    X = imputer.transform(X)

    ###Encoding the dependant variable
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    ###Splitting dataset into test set and training set
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                    test_size = 0.3, random_state = 0)
    ##Feature Scaling

    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    
    return X_train, y_train, X_test, y_test
    

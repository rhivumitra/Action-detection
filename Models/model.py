

# Training and validating the model

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import pandas as pd

from Preprocessing import create_dataset
from FeatureExtraction import X_train, y_train, X_test, y_test, dff

from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

import pickle


#Training by KNN
# classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
# classifier.fit(X_train, y_train)

#Training by DecisionTreeClassifier
# classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
# classifier.fit(X_train, y_train)

#Training by RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 19, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

#Prediction and cm
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

cm = confusion_matrix(y_test, y_pred)
print("training and validation results")
print(cm)
print(accuracy_score(y_test, y_pred))

#Save Model
save_path = "C:\Projects\Acceleration-based-activity-recognition\Saved-Models" + "\model" + ".sav"
pickle.dump(classifier, open(save_path, 'wb'))

#Testing Model
with open(save_path, "rb") as f:
    loaded_model = pickle.load(f)
    result = loaded_model.predict(create_dataset(dff)[2])
    print("testing results")
    print(confusion_matrix(create_dataset(dff)[3], result))
    print(accuracy_score(create_dataset(dff)[3], result))






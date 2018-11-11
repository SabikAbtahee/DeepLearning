#import libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




# File to train
File='C:\\Users\\sabik\\PycharmProjects\\DeepLearningBasics\\Churn_Modelling.csv'


# Dataset making
dataset=pd.read_csv(File)

X=dataset.iloc[: , 3:13].values
Y=dataset.iloc[:,13].values


# Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X=X[:, 1:]
# Train Test splitting
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.1,random_state=0)

# Feature scaling

from sklearn.preprocessing import StandardScaler
sc= StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

# importing keras
import keras
from keras.models import Sequential
from keras.layers import Dense


# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 11))

# Adding the second hidden layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  

# Fitting the ANN to the Training set
classifier.fit(X_train, Y_train, batch_size = 5, epochs = 20)


# Part 3 - Making predictions and evaluating the model

# Predicting the Test set results
Y_pred = classifier.predict(X_test)
Y_pred = (Y_pred > 0.5)

#test individual data
test_set=sc.transform(np.array([[0.0   ,0.0  , 600.0 ,  1.0 ,  40.0  ,3.0 ,60000.0 ,2.0 ,1.0, 1.0 ,50000.0]]))
test_result = classifier.predict(test_set)
test_result
test_result = (test_result >0.5)
test_result
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
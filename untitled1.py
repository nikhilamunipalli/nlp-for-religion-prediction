#importing libraries
import numpy as np
import pandas as pd
'''import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable'''

#importing datasets
dataset = pd.read_csv('train_dataset.csv')
X = dataset.iloc[:,[0,2,3,6,9]].values
y = dataset.iloc[:,10].values

#categorical data handling
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

l=[0,2]

for j in l :
     X[:,j] = labelencoder.fit_transform(X[:,j])

#standard scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

#training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2)

#classification
from sklearn.linear_model import LogisticRegression as C
classifier = C()
classifier.fit(X_train,y_train)

#predicting
y_pred = classifier.predict(X_test)

#confusionmatrix
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_pred,y_test)













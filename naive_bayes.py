#importing libraries
import numpy as np
import pandas as pd
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier


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
#building the ANN model
def build_classifier():
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim = 5 , output_dim = 50, init = 'uniform', activation = 'relu'))
    
    #internal layers
    classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 50, init = 'uniform', activation = 'relu'))
    
    #output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    
    #compiling ANN
    classifier.compile(optimizer = 'rmsprop' , metrics =['accuracy'],loss = 'binary_crossentropy' )
    return classifier

#kfoldcrossvalidaton
from sklearn.model_selection import cross_val_score as cs
classifier = KerasClassifier(build_fn = build_classifier, batch_size = 5, nb_epoch = 100)
accuracies = cs(classifier,X = X_train, y = y_train, cv = 10, n_jobs = -1)
accuracy = accuracies.mean()

#fitting the model
classifier.fit(X_train,y_train)

#predicting
y_pred = classifier.predict(X_test)

#confusionmatrix
from sklearn.metrics import confusion_matrix
cm =confusion_matrix(y_pred,y_test)

#kfoldcrossvalidation
'''from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv =10)
accuracies.mean()'''












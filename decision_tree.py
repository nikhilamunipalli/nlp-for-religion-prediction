#importing libraries
import numpy as np
import pandas as pd

#importing datasets
dataset = pd.read_csv('train_dataset.csv')
X = dataset.iloc[:,[0,1,2,3,4,6,9]].values
y = dataset.iloc[:,10].values
test = pd.read_csv('test_dataset.csv').iloc[:,[0,1,2,3,4,6,9]].values
train_length = len(X)
test_length = len(test)
X = np.append(X,test, axis =0)

#natural language processing for names
l = [1,4]
for i,j in enumerate(l) : 
    from sklearn.feature_extraction.text import CountVectorizer
    corpus = []
    features = []
    corpus = X[:,[j-i]]
    corpus = np.vstack(corpus)
    corpus.tolist()
    cv = CountVectorizer(max_features = 100)
    features = cv.fit_transform(corpus.ravel()).toarray()
    X = np.append(X,features ,axis = 1)
    X = np.delete(X,j-i,1)

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

#seperation of test and train
X_test = X[train_length:,:]
X_train = X[:train_length,:]
y_train = y

#classification
from sklearn.tree import DecisionTreeClassifier as C
classifier = C()
classifier.fit(X_train,y_train)

#predicting
y_pred = classifier.predict(X_test)

#kfoldcrossvalidation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv =10)
accuracies.mean()






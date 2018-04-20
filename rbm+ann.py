#importing libraries
import numpy as np
import pandas as pd
import keras 
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable

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

#classification
#converting to torch tensors
X = torch.FloatTensor(X)


#class RBM
class RBM():
    def __init__(self,nh,nv):
        self.W = torch.randn(nh,nv)
        self.a = torch.randn(1,nh)
        self.b = torch.randn(1,nv)
    def sample_h(self,x):
        wx = torch.mm(x,self.W.t())
        activation = wx + self.a.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)
    def sample_v(self,y):
        wy = torch.mm(y,self.W)
        activation = wy + self.b.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)
    def train(self,v0,vk,ph0,phk):
        self.W += torch.mm(v0.t(),ph0) - torch.mm(vk.t(),phk)
        self.a += torch.sum((ph0-phk),0)
        self.b += torch.sum((v0-vk),0)
        
        
#instantiation
nb_x = len(X)
nv = len(X[0])
nh = 50
batch_size = 10
rbm = RBM(nh,nv)
train = []
nb_epoch = 10

#training the rbm
for epoch in range(nb_epoch+1):
    train_loss = 0
    s = 0
    for id_x in range(0,nb_x,batch_size):
        vk = X[id_x : id_x + batch_size]
        v0 = X[id_x : id_x + batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(50):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0,vk,ph0,phk)
        p = phk.numpy()
        
        if epoch== nb_epoch and batch_size == 10:
           train.append(p)
           
        train_loss += torch.mean(torch.abs(v0 - vk))
        s+=1;
    print('epoch :'+str(epoch)+' loss : '+str(train_loss/s))

#training set conversion
train = np.vstack(train)

#training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(train,y,test_size = 0.2)

#building the ANN model
def build_classifier():
    classifier = Sequential()
    
    #first layer of neural network
    classifier.add(Dense(input_dim = nh , output_dim = 50, init = 'uniform', activation = 'relu'))
    
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











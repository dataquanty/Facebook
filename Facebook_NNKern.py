from __future__ import absolute_import
from __future__ import print_function

import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.utils import np_utils, generic_utils
from keras.optimizers import SGD
from keras.optimizers import Adagrad
from keras.regularizers import l2

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


'''
    This demonstrates how to reach a score of 0.4890 (local validation)
    on the Kaggle Otto challenge, with a deep net using Keras.
    Compatible Python 2.7-3.4 
    Recommended to run on GPU: 
        Command: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python kaggle_otto_nn.py
        On EC2 g2.2xlarge instance: 19s/epoch. 6-7 minutes total training time.
    Best validation score at epoch 21: 0.4881 
    Try it at home:
        - with/without BatchNormalization (BatchNormalization helps!)
        - with ReLU or with PReLU (PReLU helps!)
        - with smaller layers, largers layers
        - with more layers, less layers
        - with different optimizers (SGD+momentum+decay is probably better than Adam!)
    Get the data from Kaggle: https://www.kaggle.com/c/otto-group-product-classification-challenge/data
'''

#np.random.seed(1337) # for reproducibility

def load_data(path, train=True):
    df = pd.read_csv(path)
    X = df.values.copy()
    if train:
        np.random.shuffle(X) # https://youtu.be/uyUXoap67N8
        X, labels = X[:, 1:-1].astype(np.float32), X[:, -1]
        return X, labels
    else:
        X, ids = X[:, 1:].astype(np.float32), X[:, 0].astype(str)
        return X, ids

def preprocess_data(X, scaler=None):
    if not scaler:
        scaler = StandardScaler()
        scaler.fit(X)
    X = scaler.transform(X)
    return X, scaler

def preprocess_labels(labels, encoder=None, categorical=True):
    if not encoder:
        encoder = LabelEncoder()
        encoder.fit(labels)
    y = encoder.transform(labels).astype(np.int32)
    if categorical:
        y = np_utils.to_categorical(y)
    return y, encoder

def make_submission(y_prob, ids, encoder, fname):
    with open(fname, 'w') as f:
        f.write('id,')
        f.write(','.join([str(i) for i in encoder.classes_]))
        f.write('\n')
        for i, probs in zip(ids, y_prob):
            probas = ','.join([i] + [str(p) for p in probs.tolist()])
            f.write(probas)
            f.write('\n')
    print("Wrote submission to file {}.".format(fname))

def grow1(df,n_times):
    a = df[df[:,1]==1].copy()
    for i in range(n_times):
        df = np.vstack((df,a))
    
    return df


print("Loading data...")
X = pd.read_csv('train.csv').values.copy()
mat = pd.read_csv('train.csv')
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(pd.read_csv('Y.csv',header=None))

df = np.hstack((y,X))
#df = grow1(df,2)

np.random.shuffle(df)

y = df[:,:2]
X = df[:,2:]

X_test = pd.read_csv('Xtest.csv')
X_test, _ = preprocess_data(X_test, scaler)

nb_classes = y.shape[1]
print(nb_classes, 'classes')

dims = X.shape[1]
print(dims, 'dims')
print(len(X),'length')

print("Building model...")

"""
model = Sequential()
model.add(Dense(dims, 64, init='glorot_uniform'))
model.add(PReLU((64,)))
model.add(BatchNormalization((64,)))
model.add(Dropout(0.5))

model.add(Dense(64, 64, init='glorot_uniform'))
model.add(PReLU((64,)))
model.add(BatchNormalization((64,)))
model.add(Dropout(0.5))

model.add(Dense(64, 64, init='glorot_uniform'))
model.add(PReLU((64,)))
model.add(BatchNormalization((64,)))
model.add(Dropout(0.5))

model.add(Dense(64, nb_classes, init='glorot_uniform'))
model.add(Activation('softmax'))
"""

NN = 1024
dropout = 0.7
model = Sequential()


#model.add(Dense(dims, 30, W_regularizer = l1(.01)))

model.add(Dense(dims, 1024, init='glorot_uniform'))
model.add(Activation('tanh'))
model.add(PReLU((1024,)))
model.add(BatchNormalization((1024,)))
model.add(Dropout(dropout))

model.add(Dense(1024, NN, init='glorot_uniform'))
model.add(Activation('tanh'))
model.add(PReLU((NN,)))
model.add(BatchNormalization((NN,)))
model.add(Dropout(dropout))

model.add(Dense(NN, 20, init='glorot_uniform'))
model.add(Activation('tanh'))
model.add(PReLU((20,)))
model.add(BatchNormalization((20,)))
model.add(Dropout(dropout))

#model.add(Dense(5, 5, W_regularizer = l2(.5)))
model.add(Dense(20, 20, W_regularizer = l2(.01)))

model.add(Dense(20, nb_classes, init='normal'))
model.add(Activation('sigmoid'))




#sgd = SGD(lr=100*1e-3, decay=1e-5, momentum=0.9, nesterov=True)
sgd = SGD(lr=100*1e-3, decay=1e-7, momentum=0.7, nesterov=True)
adagrad = Adagrad(lr=0.1, epsilon=1e-6)


model.compile(loss='binary_crossentropy', optimizer=sgd)

print ("load weights")
#model.load_weights('weightsNN')

print("Training model...")

hist = model.fit(X, y, nb_epoch=150, batch_size=256, validation_split=0.3,show_accuracy=True,verbose=2,shuffle=True)

print("Generating submission...")

X_test = pd.read_csv('Xtest.csv')
X_test, _ = preprocess_data(X_test, scaler)

proba = model.predict_proba(X_test,verbose=2)
pd.DataFrame(proba[:,1]).to_csv('submission.csv',index=False)
print (np.sum(proba[:,1]>0.5))
#make_submission(proba, ids, encoder, fname='submission.csv')




X = pd.read_csv('train.csv').values.copy()
X, scaler = preprocess_data(X)
y, encoder = preprocess_labels(pd.read_csv('Y.csv',header=None))


proba = model.predict_proba(X,verbose=2)
pd.DataFrame(proba).to_csv('Ytrain.csv')
#make_submission(proba, ids, encoder, fname='Ytrain.csv')

print (roc_auc_score(y[:,1],proba[:,1]))


"""
sgd = SGD(lr=1e-2, decay=1e-3, momentum=0.8, nesterov=True)
model.fit(X, y, nb_epoch=100, batch_size=128, validation_split=0.2,show_accuracy=True,verbose=2,shuffle=True)
"""

"""
NN = 650
sgd = SGD(lr=1e-2, decay=1e-7, momentum=0.9, nesterov=True)
model.fit(X, y, nb_epoch=200, batch_size=1010, validation_split=0.2,show_accuracy=True,verbose=2,shuffle=True)
"""

plt.figure()
plt.subplot(1,2,1)
plt.title('Accuracy')
plt.plot(hist['acc'],'b-',label='Training Accuracy')
plt.plot(hist['val_acc'],'r-', label = 'Validation accuracy')
plt.ylim( (0.5, 1) )
plt.legend(loc='lower right')
plt.subplot(1,2,2)
plt.title('Loss')
plt.plot(hist['loss'],'b-',label='Training Loss')
plt.plot(hist['val_loss'], 'r-',label = 'Validation Loss')
plt.ylim( (0, 0.5) )
plt.legend(loc='upper right')


# -*- coding: utf-8 -*-
"""
Created on Mon May 11 18:46:30 2015

@author: dataquanty
"""

import pandas as pd
import numpy as np
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from operator import itemgetter
import matplotlib.pyplot as plt
from scipy.stats import randint as sp_randint
import scipy
from larspath import LarsPath
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale
from sklearn.svm import SVC
from vectorizer import vectorize
from sklearn.decomposition import PCA
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.utils import shuffle


mat1 = pd.read_csv('train.csv')
mat2 = pd.read_csv('test.csv')
bids1 = pd.read_csv('bidagg.csv')
bids2 = pd.read_csv('bidderagg.csv')

bids = bids1.merge(bids2,left_on='bidder_id',right_on='bidder_id')

trainLen = len(mat1)

mat = pd.concat([mat1,mat2])

#bids.drop('Unnamed: 0',axis = 1,inplace=True)
#mat['outcome']=1-mat['outcome']

mat = mat.merge(bids,how='left',left_on='bidder_id',right_on='bidder_id')

mat = mat.fillna(mat.median())
mat['address'] = mat['address'].apply(lambda x : hash(x))
mat['bidder_id'] = mat['bidder_id'].apply(lambda x : hash(x))
mat['payment_account'] = mat['payment_account'].apply(lambda x : hash(x))
mat.drop(['address','bidder_id','payment_account'],axis=1,inplace=True)

cols = [x for x in mat.columns if 'topCat' in x ]

for c in cols:
    mat[c] = mat[c].apply(lambda x : hash(x))

"""
#vectorize inputs
vectorTopCat = vectorize(mat,cols,0.05)
mat = pd.concat([mat,vectorTopCat],axis=1)
mat.drop(cols,axis=1,inplace=True)
"""

mat['dummy']=np.random.random(len(mat))


cols = [x for x in mat.columns if 'entropy2' in x]
for c in cols:
    mat[c]=np.log(mat[c]*1e6)


cols = mat.drop('outcome',axis=1).columns


for c in cols:
    try:
        mat[c+'_|16'] = pd.cut(mat[c],16,labels=False)
        mat[c+'_|32'] = pd.cut(mat[c],32,labels=False)
        mat[c+'_|64'] = pd.cut(mat[c],64,labels=False)
        mat[c+'_|128'] = pd.cut(mat[c],128,labels=False)
        mat.drop(c,axis=1,inplace=True)
    except:
        mat[c] = np.nan_to_num(mat[c])
        mat[c+'_|16'] = pd.cut(mat[c],16,labels=False)
        mat[c+'_|32'] = pd.cut(mat[c],32,labels=False)
        mat[c+'_|64'] = pd.cut(mat[c],64,labels=False)
        mat[c+'_|128'] = pd.cut(mat[c],128,labels=False)
        mat.drop(c,axis=1,inplace=True)
        print c 

cols = mat.drop('outcome',axis=1).columns
for c in cols:
    try:
        mat[c]=mat[c].astype('int8')
    except:
        print c

#mat = pd.concat([mat['outcome'],mat[featImport.index[:100]]],axis=1)


Y = np.array(mat['outcome'][0:trainLen])

X = np.array(mat.drop('outcome',axis=1)[0:trainLen])
XTestFb = np.array(mat.drop('outcome',axis=1)[trainLen:])


X, Y = shuffle(X,Y)


offset = int(X.shape[0] * 0.8)
X_train, y_train = X[:offset], Y[:offset]
X_test, y_test = X[offset:], Y[offset:]


n_est = 400
params = {'n_estimators': n_est, 
#          'max_depth': 3, 
          'min_samples_split': 4,
          'learning_rate': 0.01, 
          'subsample':0.7,
#          'max_features':'sqrt'
          }


clf = GradientBoostingClassifier(**params)

clf.fit(X_train, y_train)
roc = roc_auc_score(y_test, clf.predict(X_test))
print("ROC: %.4f" % roc)




test_score = np.zeros((params['n_estimators'],), dtype=np.float64)

for i, y_pred in enumerate(clf.staged_decision_function(X_test)):
    test_score[i] = clf.loss_(y_test, y_pred)

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title('Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, clf.train_score_, 'b-',
         label='Training Set Deviance')
plt.plot(np.arange(params['n_estimators']) + 1, test_score, 'r-',
         label='Test Set Deviance')
plt.legend(loc='upper right')
plt.xlabel('Boosting Iterations')
plt.ylabel('Deviance')



cols = np.array(mat.drop('outcome',axis=1).columns)
importance = clf.feature_importances_

#plt.figure()
featImport = pd.concat((pd.DataFrame(cols),pd.DataFrame(importance)),axis=1)
featImport.columns=['f','v']
featImport.sort('v',ascending=False,inplace=True)
featImport.set_index('f',inplace=True)

featImport.plot(kind='bar')
plt.subplots_adjust(bottom = 0.3)
plt.show()





def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")




          

paramDist = {'n_estimators': sp_randint(100,4000),
             'learning_rate': [0.05],
             'max_features':['auto'],
             'max_depth': scipy.stats.expon(scale=4),
#             'min_samples_split':scipy.stats.expon(scale=2),
             'min_samples_leaf':[1]}

Gradboost = GradientBoostingClassifier()
LogReg = LogisticRegression(class_weight='auto')
SVMCl = SVC(class_weight='auto')
metric = roc_auc_score

grid_search = RandomizedSearchCV(Gradboost,cv=3,param_distributions=paramDist,n_iter=50,n_jobs=4, scoring='precision')

grid_search.fit(X, Y)

scoresGrid = grid_search.grid_scores_
print grid_search.best_score_
print grid_search.best_estimator_
report(grid_search.grid_scores_)


cols = np.array(mat.drop('outcome',axis=1).columns)
importance = grid_search.best_estimator_.feature_importances_

#plt.figure()
featImport = pd.concat((pd.DataFrame(cols),pd.DataFrame(importance)),axis=1)
featImport.columns=['f','v']
featImport.sort('v',ascending=False,inplace=True)
featImport.set_index('f',inplace=True)

featImport.plot(kind='bar')
plt.subplots_adjust(bottom = 0.3)
plt.show()


Gradboost = GradientBoostingClassifier(n_estimators=400, learning_rate= 0.008, max_depth= 3, subsample=0.7,min_samples_split=1)
Rforest = RandomForestClassifier(class_weight='subsample', n_estimators=2000, criterion='entropy', max_depth= 50)
calibrated_clf = CalibratedClassifierCV(Rforest, method='isotonic',cv=3)
calibrated_clf = CalibratedClassifierCV(Gradboost, method='isotonic', cv=3)


calibrated_clf.fit(X, Y)
Bagging = BaggingClassifier(base_estimator=calibrated_clf,n_estimators=100,random_state=23,n_jobs=4)
Bagging.fit(X,Y)


Y2 = calibrated_clf.predict_proba(X)[:,1]
Y2= Bagging.predict_proba(X)[:,1]
Y2 = clf.predict_proba(X)[:,1]



Y2 = grid_search.best_estimator_.predict(X)
roc_auc_score(Y,Y2)

df = pd.DataFrame(np.vstack((Y,Y2)).T)
(df[df[0]!=df[1]]).shape

Ytest = calibrated_clf.predict_proba(XTest)[:,1]
Ytest = grid_search.best_estimator_.predict_proba(XTest)[:,1]
Ytest = Bagging.predict_proba(XTest)[:,1]

mat2['prediction']= np.round(Ytest,8)
mat2[['bidder_id','prediction']].to_csv('sub19-GBM-ISO-Bag.csv',index=False)


### AGG
p1 = pd.read_csv('sub19-RF-ISO-Bag.csv')
p2 = pd.read_csv('sub19-GBM-ISO-Bag.csv')
p = np.round((p1['prediction']+p2['prediction'])/2,decimals=8)

pd.concat([p1['bidder_id'],p],axis=1).to_csv('sub19-RF-GBM-ISO-Bag.csv',index=False)


"""
cols = [x for x in mat.columns if 'time_diff_cl' in x ]
cols = [x for x in cols if x not in ['time_diff_cl150_entropy','time_diff_cl150_uniqRatio']]
mat.drop(cols,axis=1,inplace=True)
"""

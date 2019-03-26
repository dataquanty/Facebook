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


mat1 = pd.read_csv('train.csv')
mat2 = pd.read_csv('test.csv')
bids = pd.read_csv('bidagg.csv')

trainLen = len(mat1)

mat = pd.concat([mat1,mat2])

bids.drop('Unnamed: 0',axis = 1,inplace=True)
mat['outcome']=1-mat['outcome']



mat = mat.merge(bids,how='left',left_on='bidder_id',right_on='bidder_id')

mat = mat.fillna(mat.median())
mat['address'] = mat['address'].apply(lambda x : hash(x))
mat['bidder_id'] = mat['bidder_id'].apply(lambda x : hash(x))
mat['payment_account'] = mat['payment_account'].apply(lambda x : hash(x))
mat.drop(['address','bidder_id','payment_account'],axis=1,inplace=True)

cols = mat.drop('outcome',axis=1).columns

for c in cols:
    mat[c] = pd.cut(mat[c],10,labels=False)

Yt = np.array(mat['outcome'][0:int(trainLen/2)])
Y = np.array(mat['outcome'][int(trainLen/2):trainLen])
Xt = np.array(mat.drop('outcome',axis=1)[0:int(trainLen/2)])
X = np.array(mat.drop('outcome',axis=1)[int(trainLen/2):trainLen])
XTest = np.array(mat.drop('outcome',axis=1)[trainLen:])

"""
lars = LarsPath(X,Y,cols)
lars.compute_path()
lars.plot_path()
X = X[:,lars.path_indices[-3:]]
"""
X = scale(X,axis=1)


def report(grid_scores, n_top=5):
    top_scores = sorted(grid_scores, key=itemgetter(1), reverse=True)[:n_top]
    for i, score in enumerate(top_scores):
        print("Model with rank: {0}".format(i + 1))
        print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
              score.mean_validation_score,
              np.std(score.cv_validation_scores)))
        print("Parameters: {0}".format(score.parameters))
        print("")



dictParam = {'n_estimators':[30] }
#dictParam.update({'criterion':['gini']})
dictParam.update({'max_features':['auto']})
dictParam.update({'max_depth':[15,20,30]})
dictParam.update({'min_samples_split':[2]})
dictParam.update({'min_samples_leaf':[1]})


paramDist = {'n_estimators': [100],
#             'criterion': ['gini'],
             'max_features':['auto'],
             'max_depth': scipy.stats.expon(scale=25),
             'min_samples_split':[2],
             'min_samples_leaf':scipy.stats.expon(scale=3)}
"""          

paramDist = {'n_estimators': sp_randint(50,100),
#             'criterion': ['gini'],
             'max_features':['auto'],
             'max_depth': scipy.stats.expon(scale=5),
#             'min_samples_split':scipy.stats.expon(scale=2),
             'min_samples_leaf':scipy.stats.expon(scale=1)}
"""
paramReg = {'penalty':['l2'],
            'C':[0.1,0.01,0.001,1]}

paramSVC = {'kernel':['rbf'],
            'C':[0.1,0.01,0.001,1]}



Rforest = RandomForestClassifier(class_weight='subsample')
Gradboost = GradientBoostingClassifier()
LogReg = LogisticRegression(class_weight='auto')
SVMCl = SVC(class_weight='auto')
metric = roc_auc_score

grid_search = GridSearchCV(SVMCl, cv=3, param_grid=paramSVC,n_jobs=4,pre_dispatch='1*n_jobs',scoring='precision')
grid_search = GridSearchCV(LogReg, cv=3, param_grid=paramReg,n_jobs=4,pre_dispatch='1*n_jobs',scoring='precision')
grid_search = RandomizedSearchCV(Rforest,cv=2,param_distributions=paramDist,n_iter=50,n_jobs=4, scoring='precision')
grid_search = RandomizedSearchCV(Gradboost,cv=3,param_distributions=paramDist,n_iter=10,n_jobs=4, scoring='precision')

grid_search.fit(X, Y)

scoresGrid = grid_search.grid_scores_
print grid_search.best_score_
print grid_search.best_estimator_
report(grid_search.grid_scores_)


cols = np.array(mat.drop('outcome',axis=1).columns)
importance = grid_search.best_estimator_.feature_importances_

plt.figure()
featImport = pd.concat((pd.DataFrame(cols),pd.DataFrame(importance)),axis=1)
featImport.columns=['f','v']
featImport.sort('v',ascending=False,inplace=True)
featImport.set_index('f',inplace=True)

featImport.plot(kind='bar')
plt.subplots_adjust(bottom = 0.3)
plt.show()

Y2 = grid_search.best_estimator_.predict(X)
roc_auc_score(Y,Y2)

df = pd.DataFrame(np.vstack((Y,Y2)).T)
(df[df[0]!=df[1]]).shape


Y3 = grid_search.best_estimator_.predict(Xt)
roc_auc_score(Yt,Y3)

# -*- coding: utf-8 -*-
"""
Created on Wed May  6 11:03:31 2015

@author: dataquanty
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis
import matplotlib.pyplot as plt


mat = pd.read_csv('bids.csv',sep=',')



mat['auction-device']=(mat['auction']+mat['device']).apply(lambda x: hash(x))
mat['auction-country']=(mat['auction']+mat['country']).apply(lambda x: hash(x))
mat['auction-ip']=(mat['auction']+mat['ip']).apply(lambda x: hash(x))
mat['auction-url']=(mat['auction']+mat['url']).apply(lambda x: hash(x))
mat['device-ip']=(mat['ip']+mat['device']).apply(lambda x: hash(x))
mat['device-country']=(mat['device']+mat['country']).apply(lambda x: hash(x))
mat['device-url']=(mat['device']+mat['url']).apply(lambda x: hash(x))
mat['device-merchandise']=(mat['device']+mat['merchandise']).apply(lambda x: hash(x))
mat['ip-url']=(mat['ip']+mat['url']).apply(lambda x: hash(x))
mat['ip-country']=(mat['ip']+mat['country']).apply(lambda x: hash(x))
mat['auction-ip1']= (mat['auction']+mat['ip'].apply(lambda x: str(x.split('.')[:1]))).apply(lambda x: hash (x))
mat['auction-ip2']= (mat['auction']+mat['ip'].apply(lambda x: str(x.split('.')[:2]))).apply(lambda x: hash (x))
mat['auction-ip3']= (mat['auction']+mat['ip'].apply(lambda x: str(x.split('.')[:3]))).apply(lambda x: hash (x))

mat.drop('merchandise',inplace=True,axis=1)

mat.sort(['bidder_id','auction','time'],inplace=True)
mat['time_shift']=mat['time'].shift(-1)
mat['bidder_id_shift']=mat['bidder_id'].shift(-1)
mat['auction_shift']=mat['auction'].shift(-1)
mat['time_shift_c']=mat['time_shift'] * ((mat['bidder_id']==mat['bidder_id_shift']) & (mat['auction']==mat['auction_shift']))
mat['time_diff_auct']=mat['time_shift_c']-mat['time']
#mat['bidFolDayAuct']=(mat['time_diff']>3*1e13)*1   # no one
mat['time_diff_auct'] = mat['time_diff_auct'].apply(lambda row : row if ((row >= 0) & (row<3.1e13)) else np.NaN)

mat.drop(['time_shift','bidder_id_shift','time_shift_c','auction_shift'],axis=1,inplace=True)


mat.sort(['bidder_id','time'],inplace=True)
mat['time_shift']=mat['time'].shift(-1)
mat['bidder_id_shift']=mat['bidder_id'].shift(-1)
mat['time_shift_c']=mat['time_shift'] * (mat['bidder_id']==mat['bidder_id_shift'])
mat['time_diff']=mat['time_shift_c']-mat['time']
mat['bidFolDay']=(mat['time_diff']>3*1e13)*1
mat['time_diff'] = mat['time_diff'].apply(lambda row : row if ((row >= 0) & (row<3.1e13)) else np.NaN)


mat.drop(['time_shift','bidder_id_shift','time_shift_c'],axis=1,inplace=True)


mat['timeofday']=mat['time'].apply(lambda x : x%((6.4)*1e13))


cols = [x for x in mat.columns if x not in ['bid_id','bidder_id']] 

for c in cols:
    if mat[c].dtype == np.dtype('object'):
        mat[c] = mat[c].apply(lambda x: hash(x))


for c in cols:
    mat[c] = pd.cut(mat[c],500,labels=False)

for c in cols:
    try:
        mat[c] = mat[c].astype('int16')
    except:
        print 'cannot convert ' + c
        pass



def uniqRatio(x):
    return np.float(np.size(np.unique(x)))/np.size(x)


def entropy(x):
    a = x.value_counts()/len(x)
    return -np.sum(a*np.log2(a))

def topCat(n):
    def topCat_(x):
        a = x.value_counts()
        a = a.index.tolist()[:n]
        a.sort()
        return str(a)
    topCat_.__name__ = 'topCat%s' % n
    return topCat_

def entropy2(lenSerie):
    def entropy2_(x):
        a = x.value_counts()/lenSerie 
        return -np.sum(a*np.log2(a))
    return entropy2_


aggfunc = {}
for c in cols:
    aggfunc[c]= [uniqRatio,entropy2(len(mat))]

bidderagg = mat.groupby('bidder_id').agg(aggfunc)

bidderagg.columns = ['_'.join(col).strip() for col in bidderagg.columns.values]
bidderagg = bidderagg.reset_index()



#bidagg1 = pd.read_csv('bidagg.csv')

bidderagg.to_csv('bidderagg.csv',index=False)




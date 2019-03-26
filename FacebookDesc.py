# -*- coding: utf-8 -*-
"""
Created on Tue May 19 21:47:32 2015

@author: dataquanty
"""

import pandas as pd
import numpy as np

mat = pd.read_csv('bids.csv',sep=',')
train = pd.read_csv('train.csv')

mat = mat.merge(train,left_on='bidder_id',right_on='bidder_id')



def topCat(n):
    def topCat_(x):
        a = x.value_counts()
        a = a.index.tolist()[:n]
        a.sort()
        return str(a)
    topCat_.__name__ = 'topCat%s' % n
    return topCat_


    


aggfunc = {'merchandise':[topCat(1),topCat(2),topCat(3)],
           'device':[topCat(1),topCat(2),topCat(3)],
           'country':[topCat(1),topCat(2),topCat(3)]}

   
stats = mat.groupby('bidder_id').agg(aggfunc)
stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
stats = stats.reset_index()

train = train.merge(stats,how='left',left_on='bidder_id',right_on='bidder_id')

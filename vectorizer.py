# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 10:31:35 2015

@author: gferry
"""

import pandas as pd
import numpy as np



# entrée => dataFrame + liste champs à vectoriser + seuil
# sortie => DataFrame vectorisé
# !!! NaN




def vectorize(DataFrame, cols, thres):
    mat = pd.DataFrame(DataFrame)
    nrows = len(mat)
    newmat = pd.DataFrame(dtype=int)    
    
    for field in cols:
        m = np.array((mat[field].value_counts()/nrows).reset_index())        
        m = np.array(filter(lambda row: row[1]>thres, m))        

        for e in m:
            newmat[field + '|' + str(e[0])] = mat[field].apply(lambda row: 1 if(row==e[0]) else 0)
        
        if float(mat[field].isnull().sum())/nrows>thres:
            newmat[field + '|NaN'] = mat[field].isnull().astype(int)
        
    print newmat.sum()     
    return newmat





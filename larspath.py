#!/usr/bin/env python
"""
@author: gferry
=====================
Lasso path using LARS
=====================

Computes Lasso Path along the regularization parameter using the LARS
algorithm

"""


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import lars_path


class LarsPath:
#column names are defined separately to be able to use np.array
    def __init__(self, X, Y, columns):
        self.X = X
        self.Y = Y
        self.columns = columns
        self.path_indices = None
        self.path_alphas = None
        
        
    def compute_path(self):
        var_alpha = np.zeros((self.X.shape[1],2))
        print("Computing regularization path using the LARS ...")
        alphas, _, coefs = lars_path(self.X, self.Y, verbose=True)
        
        k = 0
        for line in coefs:
            if sum(abs(line))>0:
                index = next((i for i, x in enumerate(line) if x), None)
                var_alpha[k]=[k, alphas[index]]
            else:
                var_alpha[k]=[k, 0]
            k += 1
        
        var_alpha = pd.DataFrame(var_alpha)
        var_alpha.columns = ['ind','alpha']
        var_alpha.sort('alpha',ascending = True,inplace = True)
        self.path_indices = np.array(var_alpha['ind'],dtype=int)
        self.path_alphas = np.array(var_alpha['alpha'])
        
    def plot_path(self):
        pos = np.arange(self.path_indices.shape[0]) + .5    
        plt.figure()
        plot = plt.subplot(111)
        plt.barh(pos,self.path_alphas, align='center')
        plot.tick_params(axis='both', labelsize=8)
        plt.yticks(pos,self.columns[self.path_indices])
        plt.xlabel('alpha value')
        plt.title('Variable Importance')
        plt.subplots_adjust(left = 0.3)
        plt.show()
            

# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:03:49 2019

@author: thisThatTheOther
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from fastaisnips import *

#get the dataset
dataset=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')
#Convert categorical data to type category

#combine test set and train set for data preprocessing
fulldataset=pd.concat([dataset,df_test],axis=0,sort=False)
train_cats(fulldataset)
#Encode categorical variables and fix missing data
X, y, nas = proc_df(fulldataset, 'SalePrice')

n_valid = 1459  # test set size
n_trn = len(X)-n_valid
raw_train, raw_valid = split_vals(dataset, n_trn)
X_train, X_test = split_vals(X, n_trn)
X_val,X_train = split_vals(X_train, 365)
y_train, y_test = split_vals(y, n_trn)
y_val, y_train = split_vals(y_train, 365)


#X_train.shape, y_train.shape, X_test.shape

import math
def rmse(x,y): return math.sqrt(((x-y)**2).mean())

def print_score(m):
    res = [rmse(m.predict(X_train), y_train), rmse(m.predict(X_valid), y_valid),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    if hasattr(m, 'oob_score_'): res.append(m.oob_score_)
    print(res)


def set_rf_samples(n):
    """ Changes Scikit learn's random forests to give each tree a random sample of
    n random rows.
    """
    forest._generate_sample_indices = (lambda rs, n_samples:
        forest.check_random_state(rs).randint(0, n_samples, n))
from sklearn.ensemble import forest
from sklearn.ensemble import RandomForestRegressor    
set_rf_samples(1000)
m = RandomForestRegressor(n_estimators=100, min_samples_leaf=3, oob_score=True,
                          max_features=0.5)
%time m.fit(X_train, y_train)

rfTestScore=m.score(X_train,y_train)
rfValScore=m.score(X_val,y_val)
rfOobScore=m.oob_score_



predictions=m.predict(X_test)
submission = pd.DataFrame({'Id':df_test['Id'],'SalePrice':predictions})

filename = 'House Prices Prediction 1.csv'

submission.to_csv(filename,index=False)

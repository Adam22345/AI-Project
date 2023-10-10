# -*- coding: utf-8 -*-
"""
Created on Mon Oct 10 15:17:12 2022

@author: saeed
"""

import os
import pandas as pd
import numpy as np

from sklearn import datasets
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt


heart = pd.read_csv (r'C:\Users\saeed\Desktop\Saeed\AI/heart.csv ' )
print (heart.head(5))

#print(heart.isnull().any())

# Strip non-numeric features from the dataframe
#df = df.select_dtypes(include=['int', 'float'])

#print to check that this has worked
print(heart[:5]) 

#collect the columns names for non-target features
result = []
for x in heart.columns:
    if x != 'age':
        result.append(x)
   
X = heart[result].values
y = heart['age'].values

#split data into testing and training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
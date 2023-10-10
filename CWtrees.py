#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 14:57:42 2022

@author: adam
"""


#imports needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold




heart = pd.read_csv (r'/Users/adam/CS_Year3/heart.csv' )
print (heart.head())


result = []
for x in heart.columns:
    if x != 'age':
        result.append(x)

X = heart[result].values
y = heart['age'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)   


treeheart = DecisionTreeClassifier(criterion = 'entropy')
treeheart.fit(X_train,y_train)


y_pred = treeheart.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))



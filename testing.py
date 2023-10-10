#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 19:16:29 2022

@author: adam
"""

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




data = pd.read_csv (r'/Users/adam/CSYear3/Artifical Intellgence Coursework/data.csv' )
print (data.head())


result = []
for x in data.columns:
    if x != 'Bankrupt?':
        result.append(x)

X = data[result].values
y = data['Bankrupt?'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)   


treedata = DecisionTreeClassifier(criterion = 'entropy')
treedata.fit(X_train,y_train)


y_pred = treedata.predict(X_test)

print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
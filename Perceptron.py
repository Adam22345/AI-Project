#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 17:26:49 2022

@author: adam
"""

#imports needed for the program
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
from sklearn import datasets
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler

#loading the dataset by using the path of the dataset
data = pd.read_csv (r'/Users/adam/CSYear3/Artifical Intellgence Coursework/data.csv' )
print (data.head())


result = []
for x in data.columns:
    if x != 'Bankrupt?':
        result.append(x)

#Assigning X and Y values 
X = data[result].values
y = data['Bankrupt?'].values

#Splitting the dataset with values x and y 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25) 

#This line sets up the Perceptron model
ppnData = Perceptron(max_iter=50,tol=0.001,eta0=1)  

#uses data trees to train the dataset
treedata = DecisionTreeClassifier(criterion = 'entropy')
treedata.fit(X_train,y_train)



#This part makes the prediction of the dataset 
y_pred = treedata.predict(X_test)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))



#K fold is used for cross validation 
kf = KFold(10, shuffle=True)




print("")
print("Without standardisation")
#Does a fold without standardisation
fold = 1

for train_index, validate_index in kf.split(X,y):
    ppnData.fit(X[train_index],y[train_index])
    y_test = y[validate_index]
    y_pred = ppnData.predict(X[validate_index])
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 5


print("")
print("With standardisation")


#Does a fold with standardisation
sc = StandardScaler()


fold = 1
for train_index, validate_index in kf.split(X,y):
    sc.fit(X[train_index])
    X_train_std = sc.transform(X[train_index])
    X_test_std = sc.transform(X[validate_index])
    ppnData.fit(X_train_std,y[train_index])
    y_test = y[validate_index]
    y_pred = ppnData.predict(X_test_std)
    print(f"Fold #{fold}, Training Size: {len(X[train_index])}, Validation Size: {len(X[validate_index])}")
    print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))
    fold += 3










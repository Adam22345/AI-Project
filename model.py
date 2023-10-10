#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: adam
"""


#imports needed for the program
import pandas as pd
import numpy as np
import os
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split



dataset = "." 
#loading the dataset by using the path of the dataset
filename_read = os.path.join(dataset,r'/Users/adam/CSYear3/Artifical Intellgence Coursework/data.csv')
#Edit path to when using on diffrent machine 
dataframe = pd.read_csv(filename_read)

print(dataframe[:10])
#Start row of the dataset
output = []
for x in dataframe.columns:
    if x != 'Bankrupt?':
        output.append(x)
#Outputs results of dataset 
print(output)


#Assigning X and Y values     
X = dataframe[output].values
y = dataframe['Bankrupt?'].values

#Prints X and Y values 
print(X[0:5])
print(y[0:5])
print(X.shape)

#X and Y are split into training and testing
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 

print(X_train.shape)

#k Nearest Neighbour Model 
knn_model = KNeighborsClassifier(n_neighbors=12)

#testing k Nearest Neighbour 
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)
print('kNN Accuracy: %.3f' % accuracy_score(y_test, y_pred_knn))

#Imports of the Graphs 
import matplotlib.pyplot as plt

from matplotlib import style
with plt.style.context('dark_background'):
    plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')
figure2 = plt.figure(figsize=(30,30))


#Random Forest model
accuracy_data = []
#When Adjusting the range parameters to a value of 1, 35 instead if 100 it can decrease time take for program to run. 
nums = []
for i in range(1,100):
    random_Forest_model = RandomForestClassifier(n_estimators=i, criterion='gini')
    random_Forest_model.fit(X_train, y_train)
    y_model = random_Forest_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_model)
    accuracy_data.append(accuracy)
    nums.append(i)
#Outputs the accuracy data    
print(accuracy_data)
plt.plot(nums,accuracy_data)
plt.xlabel("no. of trees (n_estimators)")
plt.ylabel("accuracy:")
plt.show()



##################################################################################


with plt.style.context('dark_background'):
    plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')
  
plt.show()
style.use('fivethirtyeight')

figure1 = plt.figure(figsize=(30,30))

#k Nearest Neighbour model
accuracy_data = []
nums = []
for i in range(1,100):
    k_NearestNeighbour_model = KNeighborsClassifier(n_neighbors=i)
    k_NearestNeighbour_model.fit(X_train, y_train)
    y_model = k_NearestNeighbour_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_model)
    accuracy_data.append(accuracy)
    nums.append(i)
    
print(accuracy_data)
print(nums)
#Outputs the accuracy data       
plt.plot(nums,accuracy_data)
plt.xlabel("no. of neighbours")
plt.ylabel("accuracy:")
plt.show()
    


#testing for Random Forest model
random_Forest_model = RandomForestClassifier(n_estimators = 26, criterion='entropy')
random_Forest_model.fit(X_train, y_train)
y_predict_random_Forest = random_Forest_model.predict(X_test)
print('RF Accuracy: %.3f' % accuracy_score(y_test, y_predict_random_Forest))


#testing for k Nearest Neighbour
k_NearestNeighbour_model = KNeighborsClassifier(n_neighbors=15)
k_NearestNeighbour_model.fit(X_train, y_train)
y_predict_k_NearestNeighbour = knn_model.predict(X_test)
print('kNN Accuracy: %.3f' % accuracy_score(y_test, y_predict_k_NearestNeighbour))

#testing for Gaussian Naive Bayes
gaussian_Naive_Bayes_model = GaussianNB()
gaussian_Naive_Bayes_model.fit(X_train, y_train)
y_predict_gaussian_Naive_Bayes = gaussian_Naive_Bayes_model.predict(X_test)
print('GNB Accuracy: %.3f' % accuracy_score(y_test, y_predict_gaussian_Naive_Bayes))







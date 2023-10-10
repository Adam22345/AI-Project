# -*- coding: utf-8 -*-
"""
Created on Wed Dec 21 15:14:01 2022
@author: Tayo, Saeed, Adam, Humzah
"""

#imports needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os
import requests
import tensorflow.keras.utils
import tensorflow as tf

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
from sklearn import preprocessing
from tensorflow import keras
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report
from matplotlib import style



#Funcion that creates a logisticRegression
def logisticRegression(X_train,X_test,y_train,y_test):
    pipe = make_pipeline(StandardScaler(), LogisticRegression())
    # apply scaling on training data
    pipe.fit(X_train, y_train) 
    # apply scaling on testing data, without leaking training data.
    pipe.score(X_test, y_test) 
    print("Logistic Regression\n")
    logreg = LogisticRegression(fit_intercept=True)
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print(classification_report(y_test, y_pred))
    print('Logistic regression Accuracy: %.3f\n' % accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=logreg.classes_)
    display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
    display.plot()
    display.ax_.set_title("LOGISTIC REGRESSION")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.gcf().axes[0].tick_params()
    plt.gcf().axes[1].tick_params()
    plt.show()
    
#a function to plot confusion matrices
def plot_confusion_matrix(cm, names, title='Confusion matrix', cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar(fraction=0.05)
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, names, rotation=45)
    plt.yticks(tick_marks, names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

#A function that creates a decision tree model
def decisionTree(X_train,X_test,y_train,y_test):
    print("DECISION TREE\n")
    #build a decision tree and fit to the training data
    treeBank = DecisionTreeClassifier(criterion = 'entropy')
    treeBank.fit(X_train,y_train)
    
    #predict values for the testing data
    y_pred = treeBank.predict(X_test)
    
    #use the confusion matrix function to build a confusion matrix
    #(twice) once with numbers, once with proportions
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

    np.set_printoptions(precision=2)
    print('CONFUSION MATRIX, WITHOUT STANDARD NORMALIZATION\n')
    print(cm)
    
    #give a grpahical representation
    plt.figure()
    plot_confusion_matrix(cm,names=(["Not-Bankrupt","Bankrupt"]), title='confusion matrix')

    ##normalised
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    print('NORMALIZED CONFUSION MATRIX\n')
    print(cm_normalized)
    
    print(classification_report(y_test, y_pred))
    #print accuracy (other metrics are available)
    print('Decision Tree Accuracy: %.2f\n' % accuracy_score(y_test, y_pred))
    
    plt.figure()
    plot_confusion_matrix(cm_normalized, names=(["Not-Bankrupt","Bankrupt"]), title='Normalized confusion matrix')
    plt.show()

    ## or using the sklearn
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=treeBank.classes_)
    disp.plot()
    disp.ax_.set_title("DECISION TREE")
    plt.xlabel("PREDICTED LABEL")
    plt.ylabel("TRUE LABEL")
    plt.gcf().axes[0].tick_params()
    plt.gcf().axes[1].tick_params()
    plt.show()

#A function that creates a neural network model
def neuralNetwork(X_train,X_test,y_train,y_test):
    print("NEURAL NETWORK\n")
    #set up a keras model.  
    model = Sequential()
    #For the model, these settings provide good accuracy with little loss
    #with 'relu' activation if x is <= 0 then x would be set to 0
    #if x is > 0 then just take the value of x
    model.add(Dense(64, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    #sigmoid activation helps with binary classification as it maps to 0 or 1 
    #which is useful given that we want to calculate a true or false outcome with our model
    model.add(Dense(1, activation='sigmoid'))
    model.add(Dense(y.shape[1],activation='softmax'))
    #compile the model setting the loss (error) measure and the optimizer
    opt = keras.optimizers.SGD(learning_rate=0.01)
    model.compile(loss='categorical_crossentropy', optimizer=opt)
    model.compile(loss='mean_squared_error', optimizer='adam')
    model.summary()

    model.fit(X_train,y_train,verbose=2,epochs=5)

    #make predictions (will give a probability distribution)
    pred = model.predict(X_test)

    #now pick the most likely outcome
    pred = np.argmax(pred,axis=1)
    y_compare = np.argmax(y_test,axis=1) 
    #and calculate accuracy
    score = metrics.accuracy_score(y_compare, pred)
    print("Neural Network Accuracy: {}\n".format(score))
    
#A function that creates a Random Forest model
def randomForest(X_train,X_test,y_train,y_test):
    print("RANDOM FOREST\n")
    
    
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
    print(classification_report(y_test, y_model))
    print('Random Forest Accuracy: %.3f' % accuracy_score(y_test, y_model))

    
    plt.plot(nums,accuracy_data)
    plt.xlabel("no. of trees (n_estimators)")
    plt.ylabel("accuracy:")
    plt.show()
    
#A function that creates a kNearestNeighbout model    
def kNearestNeighbour(X_train,X_test,y_train,y_test):
    print("K NEAREST NEIGHBOUR\n")
    
    with plt.style.context('dark_background'):
        plt.plot(np.sin(np.linspace(0, 2 * np.pi)), 'r-o')
      
    plt.show()
    style.use('fivethirtyeight')

    figure1 = plt.figure(figsize=(30,30))
    
    
    #k Nearest Neighbour Model     
    accuracy_data = []
    nums = []
    for i in range(1,100):
        k_NearestNeighbour_model = KNeighborsClassifier(n_neighbors=i)
        k_NearestNeighbour_model.fit(X_train, y_train)
        y_model = k_NearestNeighbour_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_model)
        accuracy_data.append(accuracy)
        nums.append(i)

    #Outputs the accuracy data               
    print(classification_report(y_test, y_model))
    
    print('k Nearest Neighbour Accuracy: %.3f\n' % accuracy_score(y_test, y_model))

    plt.plot(nums,accuracy_data)
    plt.xlabel("no. of neighbours")
    plt.ylabel("accuracy:")
    plt.show()
    
#A function that creates a gaussianNaiveBayes model   
def gaussianNaiveBayes(X_train,X_test,y_train,y_test):
    print("GAUSSIAN NAIVE BAYES\n")
    gaus = GaussianNB()
    gaus.fit(X_train, y_train)
    y_pred = gaus.predict(X_test)
    gausCM = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix\n")
    print(gausCM)
    print(classification_report(y_test, y_pred))
    print('Gaussian Naive Bayes Accuracy: %.3f\n' % accuracy_score(y_test, y_pred))
    cm = confusion_matrix(y_test, y_pred, labels= gaus.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=gaus.classes_)
    disp.plot()
    disp.ax_.set_title("GAUSSIAN NAIVE BAYES")
    plt.xlabel("PREDICTED LABEL")
    plt.ylabel("TRUE LABEL")
    plt.gcf().axes[0].tick_params()
    plt.gcf().axes[1].tick_params()
    plt.show()
    

dataset = "."


filename = os.path.join(dataset,'data.csv')

# data frame
df = pd.read_csv(filename)

output = []


for x in df.columns:
    if x != 'Bankrupt?':
        output.append(x)
        
#print(output)

X = df[output].values
y = df['Bankrupt?'].values



#Split the x and y into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=42)

logisticRegression(X_train, X_test, y_train, y_test)
gaussianNaiveBayes(X_train, X_test, y_train, y_test)

#use the keras built-in to ensure the targets are categories
y = keras.utils.to_categorical(y)

#Split the x and y into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=42)


  
#Splitting the x and y into training and testing specifically for the models below
decisionTree(X_train, X_test, y_train, y_test)
neuralNetwork(X_train,X_test,y_train,y_test)
randomForest(X_train, X_test, y_train, y_test)
kNearestNeighbour(X_train, X_test, y_train, y_test)

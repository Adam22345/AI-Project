# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 13:09:26 2022

@author: TDAF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import os


from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn import metrics
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
from sklearn import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report


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
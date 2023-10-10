# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:04:39 2022

@author: TDAF
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn import datasets
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification



dataset = "."


filename = os.path.join(dataset,'data.csv')

data = pd.read_csv (filename)
print (data.head())


result = []
for x in data.columns:
    if x != 'Bankrupt?':
        result.append(x)

X = data[result].values
y = data['Bankrupt?'].values

#print ("SPLITTING DATA")

X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)   
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train)  # apply scaling on training data


#print("PIPE SCORE TEST")
pipe.score(X_test, y_test)  # apply scaling on testing data, without leaking training data.


print("Logistic Regression\n")
logreg = LogisticRegression(fit_intercept=True)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)
cnf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix")
print(cnf_matrix)
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=logreg.classes_)
display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=logreg.classes_)
display.plot()
display.ax_.set_title("Logistic Regression")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.gcf().axes[0].tick_params()
plt.gcf().axes[1].tick_params()
plt.show()
"""
Created on Thu Nov 17 14:57:42 2022
@author: adam
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


dataset = "."


filename = os.path.join(dataset,r'\Users\saeed\Desktop\Saeed\AI\data.csv')

# data frame
df = pd.read_csv(filename)

output = []


for x in df.columns:
    if x != 'Bankrupt?':
        output.append(x)
        
#print(output)

X = df[output].values
y = df['Bankrupt?'].values

#use the keras built-in to ensure the targets are categories
y = keras.utils.to_categorical(y)

#Split the x and y into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=42)

######################################################### Start of Confusion Matrix

#build a decision tree and fit to the training data
treeBank = DecisionTreeClassifier(criterion = 'entropy')
treeBank.fit(X_train,y_train)

#predict values for the testing data
y_pred = treeBank.predict(X_test)
#print(y_test)
#print(t_pred)

#print accuracy (other metrics are available)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#use the confusion matrix function to build a confusion matrix
#(twice) once with numbers, once with proportions
cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))

np.set_printoptions(precision=2)
print('Confusion matrix, without normalization')
print(cm)
#give a grpahical representation
plt.figure()
plot_confusion_matrix(cm,names=(["Not-Bankrupt","Bankrupt"]), title='confusion matrix')

##normalised
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized, names=(["Not-Bankrupt","Bankrupt"]), title='Normalized confusion matrix')
plt.show()

## or using the sklearn
disp = ConfusionMatrixDisplay(confusion_matrix=cm_normalized, display_labels=treeBank.classes_)
disp.plot()
plt.show()

################################################## End of Confusion Matrix code

# prints the x training values
print(X_train.shape)

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

model.fit(X_train,y_train,verbose=2,epochs=250)

#make predictions (will give a probability distribution)
pred = model.predict(X_test)

#print(pred[0:10])

#now pick the most likely outcome
pred = np.argmax(pred,axis=1)
y_compare = np.argmax(y_test,axis=1) 
#and calculate accuracy
score = metrics.accuracy_score(y_compare, pred)
print("Accuracy score: {}".format(score))


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

#print(pred[4])
    



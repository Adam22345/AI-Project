
"""
Confusion Matrix
@author: Humzah
"""


##Author Tayo


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


from sklearn.datasets import load_digits

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import ConfusionMatrixDisplay

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow.keras.utils
from sklearn import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras import regularizers
import io
import os
import requests
from sklearn import metrics
from sklearn.model_selection import train_test_split

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


filename = os.path.join(dataset,r'\Users\44784\Desktop\IntroToAI\CW\data.csv')

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

#print(pred[4])
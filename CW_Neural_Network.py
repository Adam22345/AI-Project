##Author Tayo



from sklearn.datasets import load_digits

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

#use the keras built-in to ensure the targets are categories
y = keras.utils.to_categorical(y)

#Split the x and y into training and testing
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.25, random_state=42)

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

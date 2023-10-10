
"""
@author: Humzah
"""

import pandas as pd
import numpy as np
import os

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


#method to plot confusion matrices
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
#loading the dataset by using the path of the dataset
filename_read = os.path.join(dataset,r'\Users\44784\Desktop\IntroToAI\CW\data.csv')

#put the dataset into a DataFrame
df = pd.read_csv(filename_read)

#shuffle the data
df = df.reindex(np.random.permutation(df.index))

#print(df.head())

result = []
for x in df.columns:
    if x != 'Bankrupt?':
        result.append(x)

#Define X and y        
X = df[result].values
y = df['Bankrupt?'].values

#Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=42) 

#print(df.columns)

#build a multiclass SVM 'ovo' for one-versus-one, and
#fit the data
multi_svm = SVC(gamma='scale', decision_function_shape='ovo')  
multi_svm.fit(X_train,y_train)

y_pred = multi_svm.predict(X_test)

#put the results into a DataFrame and print side-by-side
output = pd.DataFrame(data=np.c_[y_test,y_pred])
print(output)

#calculate accuracy score and print
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#find the confusion matrix, normalise and print
cm = confusion_matrix(y_test, y_pred)
np.set_printoptions(precision=2)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
print('Normalized confusion matrix')
print(cm_normalized)

#confusion matric as a figure
plt.figure()
plot_confusion_matrix(cm_normalized, names=(["Bankrupt","predict-Bankrupt"]), title='Normalized confusion matrix')
plt.show()

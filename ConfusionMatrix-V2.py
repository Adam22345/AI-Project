
"""
Confusion Matrix
@author: Humzah
"""

#imports for the problem
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay

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

#perform a single split, with 25% of the data for testing
X_train, X_test, y_train, y_test = train_test_split(    
    X, y, test_size=0.25, random_state=100) 

#build a decision tree and fit to the training data
treeBank = DecisionTreeClassifier(criterion = 'entropy')
treeBank.fit(X_train,y_train)

#predict values for the testing data
y_pred = treeBank.predict(X_test)
#print(y_test)
#print(y_pred)

#print accuracy (other metrics are available)
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

#use the confusion matrix function to build a confusion matrix
#(twice) once with numbers, once with proportions
cm = confusion_matrix(y_test, y_pred)
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


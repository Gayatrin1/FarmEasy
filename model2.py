# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.metrics import classification_report
import seaborn as sns

df = pd.read_csv("C:/Users/Gayatri/Desktop/py4e/FarmEasy/cropdataNew3.csv")

xtrain = df.iloc[:2880, :7].values    
ytrain = df.iloc[:2880, -1].values
xtest = df.iloc[2880:, :7].values
ytest = df.iloc[2880:, -1].values
acc = []
model = []

from sklearn.ensemble import RandomForestClassifier
#Fitting model with trainig data
from sklearn.metrics import recall_score, accuracy_score, confusion_matrix
rfc = RandomForestClassifier( )
rfc.fit(xtrain, ytrain)
predicted_values=rfc.predict(xtest)
x = metrics.accuracy_score(ytest,predicted_values)
acc.append(x)
#model.append('rfc')
print("RF's Accuracy is: ", x)
# Saving model to disk
pickle.dump(rfc, open('model2.pkl','wb'))
# Loading model to compare the results
model = pickle.load(open('model2.pkl','rb'))
print(classification_report(ytest,predicted_values))

#Decision tree
from sklearn.tree import DecisionTreeClassifier

DecisionTree = DecisionTreeClassifier(criterion="entropy",random_state=2,max_depth=5)

DecisionTree.fit(xtrain,ytrain)

predicted_values1 = DecisionTree.predict(xtest)
x1 = metrics.accuracy_score(ytest, predicted_values1)
acc.append(x1)
#model.append('DecisionTree')
print("DecisionTrees's Accuracy is: ", x)

print(classification_report(ytest,predicted_values1))

#SVM
from sklearn.svm import SVC
# data normalization with sklearn
from sklearn.preprocessing import MinMaxScaler
# fit scaler on training data
norm = MinMaxScaler().fit(xtrain)
X_train_norm = norm.transform(xtrain)
# transform testing dataabs
X_test_norm = norm.transform(xtest)
SVM = SVC(kernel='poly', degree=3, C=1)
SVM.fit(X_train_norm,ytrain)
predicted_values2 = SVM.predict(X_test_norm)
x2 = metrics.accuracy_score(ytest, predicted_values2)
acc.append(x2)
#model.append('SVM')
print("SVM's Accuracy is: ", x)

print(classification_report(ytest,predicted_values2))

print(df['label'].unique())
print(sns.heatmap(df.corr(),annot=True))
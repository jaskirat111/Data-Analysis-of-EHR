# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 12:43:59 2018

@author: DELL
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#ading of data
dataset=pd.read_csv('C:\\Users\\DELL\\Desktop\\ml\\naive_byes\\csv_files\\Social_Network_Ads.csv')

X=dataset.iloc[:, [2,3]].values
Y=dataset.iloc[:, 4].values  
#s[litting dataset into the training set and test set
from sklearn.cross_validation import train_test_split
           
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
#feature scaling
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)
#fitting Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(X_train,y_train)
#predicting the test resuls
y_pred=classifier.predict(X_test)
#the confusion matrix
from sklearn.metrics import confusion_matrix
std=confusion_matrix(y_test,y_pred)
#visualising the test results
from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:, 0].max()+1,step=0.01),np.arange(start=X_set[:, 1].min()-1,stop=X_set[:, 1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('SVM(Test set)')
plt.xlabel('age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()

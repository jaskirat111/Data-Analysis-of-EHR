# -*- coding: utf-8 -*-
"""
Created on Fri Jun 29 09:18:48 2018

@author: Mohit Pachauri
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#reading of data
dataset=pd.read_csv('C:\\Users\\jaskirat\\Desktop\\ml\\DataAnalysisOfEhr\\SVM\\csv_files\\Social_Network_Ads.csv')


X=dataset.iloc[:, [2,3]].values
Y=dataset.iloc[:, 4].values  


from sklearn.cross_validation import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=0)
#feature scaling for missing values
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)

#Fitting knn to the training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5,metric = 'minkowski',p = 2)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)





#predicting the test resuls


from matplotlib.colors import ListedColormap
X_set,y_set=X_test,y_test
X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min()-1,stop=X_set[:, 0].max()+1,step=0.01),np.arange(start=X_set[:, 1].min()-1,stop=X_set[:, 1].max()+1,step=0.01))
plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title('Logistic regression(Test set)')
plt.xlabel('age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()
#the confusion matrix

#visualising the test results
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 09:43:22 2018

@author: DELL
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\DELL\\Desktop\\ml\\Decision_tree_regression\\csv_files\\Position_Salaries (2).csv")
x = data.iloc[:,1:2].values

y = data.iloc[:,2].values

#fitting decision tree regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)


#predicting a new result

y_pred=regressor.predict(7)
x_grid = np.arange(min(x),max(x),0.01)
x_grid = x_grid.reshape((len(x_grid),1))

plt.scatter(x,y,color = 'blue')
plt.plot(x_grid,regressor.predict(x_grid),color='red')

plt.title('Truth or Bluff(Decision Tree Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

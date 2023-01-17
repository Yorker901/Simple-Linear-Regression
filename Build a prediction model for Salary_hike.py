# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:36:15 2022

@author: Mohd Ariz Khan
"""   
# step1: import the data files and libraires
import pandas as pd
df = pd.read_csv("Salary_Data.csv")
df.shape
df.head()

# step2: split the Variables in  X and Y's

X = df[["YearsExperience"]] # independent variable
Y = df["Salary"] # Target variable

# scatter plot between each x and Y
import matplotlib.pyplot as plt
plt.scatter(df.iloc[:,0], Y, color='red')
plt.ylabel("Salary")
plt.xlabel("YearsExperience")
plt.show()

# correlation
df.corr()

# Model fitting  --> Scikit learn
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

LR = LinearRegression()
LR.fit(X,Y)
Y_pred = LR.predict(X)
mse= mean_squared_error(Y,Y_pred)
RMSE = np.sqrt(mse)
print("Root mean squarred error: ", RMSE.round(3))
print("Rsquare: ",r2_score(Y,Y_pred).round(3)*100)

# Rsquare value is above than 90%.
# So, our model is Excellent.
#=========================================================    










# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 19:11:31 2022

@author: Mohd Ariz Khan
"""
# Simple linear regression

# import the data files and libraries
import pandas as pd
df = pd.read_csv("delivery_time.csv")
df.shape
df

# split the variables in X and Y's
x = df[["Sorting Time"]]
y = df[["Delivery Time"]]

# EDA
# scatter Plot
import matplotlib.pyplot as plt
plt.scatter(x = df["Sorting Time"], y = df["Delivery Time"], color="red")
plt.show()
#===============================================================

# Model fitting --> Scikitlearn
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x,y)
y_pred = LR.predict(x)
y_pred

import matplotlib.pyplot as plt
plt.scatter(x, y, color="red")
plt.plot(x, y_pred, color="blue")
plt.show()
#=============================================================

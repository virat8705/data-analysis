# -*- coding: utf-8 -*-
"""
Created on Wed Sep 20 13:33:05 2023

@author: virat
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('50_Startups.csv')
x = data.iloc[:,:-1]
y = data.iloc[:,4]

states = pd.get_dummies(x['State'],drop_first=True)

x = x.drop('State',axis=1)

x= pd.concat([x,states],axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
MLR = LinearRegression()
MLR.fit(x_train,y_train)

y_predict = MLR.predict(x_test)

from sklearn.metrics import r2_score
score = r2_score(y_test,y_predict)

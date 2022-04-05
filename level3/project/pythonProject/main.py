from sklearn.preprocessing import LabelEncoder
from regression.linearMl import Linear
from regression.linear_with_multi_vars  import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from regression.poly_regression import PloyRegr
from sklearn import *
import joblib

# data=pd.read_csv('https://raw.githubusercontent.com/codebasics/py/master/ML/5_one_hot_encoding/Exercise/carprices.csv')
# print(data.head)
# le=LabelEncoder()
# datacopy=data
# datacopy['Car Model'] =le.fit_transform(datacopy['Car Model'])
# print(datacopy)
# X=datacopy[['Car Model','Mileage','Age(yrs)']]
# Y=datacopy['Sell Price($)']
# model=linear_model.LinearRegression()
# model.fit(X,Y)
# print(model.coef_)
# print(model.predict([[1,69000,6]]))
# print( model.score(X,Y))
# joblib.dump(model,'carPrice')
# x=linear_model.LinearRegression()
# x..score
# model=joblib.load('carPrice')
# print(print(model.predict([[1,69000,6]])))
# Y=data['SalePrice']
# model= linear_model.LinearRegression()
# model.fit(X,Y)
# print(model.coef_)
# print(model.predict([[8000,2000,200]])
# model= joblib.load('houseModel')
# print(model.predict([[0,0,0]]))
x=[3,21,22,34,54,34,55,67,89,99]
y=[1,10,14,34,44,36,22,67,79,90]
x=np.array(x)
y=np.array(y)
liner=Linear(x,y,0,0)
liner.DoLinear()


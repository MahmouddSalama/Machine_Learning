import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.preprocessing import PolynomialFeatures
import seaborn as sns
import matplotlib.pyplot as lt


# read the data form ffile
data=pd.read_csv(r'assignment2\assignment2_dataset_cars.csv')
print(data.head()) # print the head

print(data.isnull().sum()) # check if the data has the null values 

le=LabelEncoder() # 
data['car_maker']=le.fit_transform(data['car_maker']) # fit the maket to int values
print(data.head()) 

# show the corr matrix in data dependent attripute 
sns.set_context('paper',font_scale=1.4) 
sns.heatmap(data.corr(),annot=True)
lt.show()

# split the data to tran & test in 80|20
xtran,xtest,ytran,ytest=train_test_split(data.drop(['price'],axis=1),data['price'],train_size=.8) 

# Polynomial model to learn in degree 5
# if degree > 5 over fit and the test error biger
poly=PolynomialFeatures(degree=5)
xpoly=poly.fit_transform(xtran)
linear=LinearRegression()
linear.fit(xpoly,ytran)
linear.score(poly.fit_transform(xtest),ytest)

# score & mean_squared_error
pred=linear.predict(poly.fit_transform(xtest))
print("the mean  square error =  " +str( metrics.mean_squared_error(ytest,pred)))
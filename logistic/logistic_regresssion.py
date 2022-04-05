import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import scale

df=pd.read_csv('conda-meta\data\insurance_data.csv')
print(df.head())
x=df[['age']] 
y=df['bought_insurance']
plt.scatter(x,y)
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.9) 

model= LogisticRegression()
model.fit(X_train,y_train)
print(model.predict(X_test))
print(model.score(X_test,y_test))


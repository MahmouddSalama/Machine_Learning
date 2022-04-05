from sklearn import linear_model

class LinearWithMultiVars:
    regr=linear_model.LinearRegression()
    def __init__(self,x=[],y=[]):
        self.x=x
        self.y=y

    def doLearn(self):
        self.regr.fit(self.x,self.y)
    
    def predict(self,data):
        return self.regr.predict(data)
    
    def coef(self):
        self.doLearn()
        return self.regr.coef_




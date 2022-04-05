
import matplotlib.pyplot as plt
import numpy as np

class Linear:
    Lr = 1e-4

    def __init__(self, x, y, a, b):
        self.x = x
        self.y = y
        self.a = a
        self.b = b

    def model(self):
        return self.a * self.x + self.b

    def costFunction(self):
        l = len(self.x);
        predection = self.model();
        return 1 / (2 * l) * (np.square(self.predection - self.y)).sum()

    def miniError(self):
        l = len(self.x)
        predection = self.model()
        da = (1 / l) * ((predection - self.y) * self.x).sum()
        db = (1 / l) * (predection - self.y).sum()
        self.a = self.a - self.Lr * da
        self.b = self.b - self.Lr - db
        return self.a, self.b

    def iteritor(self):
        for i in range(50000):
            a, b = self.miniError()
        return a, b

    def DoLinear(self):
        a, b = self.iteritor()
        predection = self.model()
        #loss = self.costFunction()
        plt.scatter(self.x, self.y,color='red',marker='*')
        plt.plot(self.x, predection,color='black',)
        plt.show()

    def newdata(self ,x):
        for i in x:
          print(self.a * i + self.b);

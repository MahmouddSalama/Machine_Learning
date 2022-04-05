import numpy as np
import matplotlib.pyplot as plt


class PloyRegr:
    x = [1, 2, 3, 5, 6, 7, 8, 9, 10, 12, 13, 14, 15, 16, 18, 19, 21, 22]
    y = [100, 90, 80, 60, 60, 55, 60, 65, 70, 70, 75, 76, 78, 79, 90, 99, 99, 100]
    def __init__(self, dataX, dataY):
        self.dataX = dataX,
        self.dataY = dataY

    def showPoints(self):
        plt.scatter(self.dataX, self.dataY)
        plt.show()

    def showgraph(self):
        myModel = np.poly1d(np.polyfit(self.dataX,self.dataY,3))
        myLine = np.linspace(1, 20, 100)
        plt.scatter(self.dataX, self.dataY)
        plt.plot(myLine, myModel(myLine))
        plt.show()
        return


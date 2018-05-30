import numpy as np
import math


x = [.1,.3,.4,.8,.9]
y = [10,8.2,4.3,2.6,.9]

Z = np.vstack((x,y))

xy = [0, 0, 0, 0, 0, 0, 0, 0, 0]



def Var(x):
    xBar = np.mean(x)

    _sum = 0.0

    for i in range(len(x)):
        _sum += (x[i] - xBar)**2
        
    var = _sum/(len(x)-1)

    return var


def Cov(x,y):
    xBar = np.mean(x)
    yBar = np.mean(y)

    _sum = 0.0

    for i in range(len(x)):
        _sum += ((x[i]-xBar) * (y[i]-yBar))
        
    cov = _sum / (len(x)-1)

    return cov
        

#def CovMatrix(Z):


xBar = np.mean(x)
yBar = np.mean(y)

xVar = Var(x)
yVar = Var(y)



print("xBar: ", xBar)
print("yBar: ", yBar)
print("xVar_1: ", xVar)
print("yVar_1: ", yVar)
print("cov_sample: ", Cov(x,y))
print()
print("np.cov: ", np.cov(x,y)[0,1])
print("np.cov matrix:")
print(np.cov(Z))
print()

r = Cov(x,y)/(math.sqrt(xVar)*math.sqrt(yVar))
#print(r)













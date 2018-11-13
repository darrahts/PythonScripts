import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import curve_fit

def F(x,a,b,c):
    return a*np.exp(-b*x) + c


x = np.array([44100, 44500, 50200, 62000, 57000, 46680, 20900, 25400, 35110, 40200, 33800, 29800, 26880, 26880,
     12500, 13900, 14100, 16600, 13900, 16100, 19800, 24900, 22800, 20400, 11900, 18760, 22500, 41000, 38200,
     45500, 45900, 20680, 38500, 31800, 31500, 26800, 23300, 20900, 12500, 25800, 20400, 20100, 17900, 16850,
     16100, 15090, 12500, 13200, 14200, 12300, 13200, 11600, 10800, 10200, 9800, 8400, 11450, 10800, 9000, 6800,
     5750, 5135, 7500, 8750, 10300, 12250, 14090, 17600, 15000, 13700, 11500, 10350, 8600, 7700, 5400, 5600, 8450])

y = np.array([5, 5.25, 4.75, 4.25, 4.5, 5, 7.3, 6.6, 5.4, 5.75, 6, 6.25, 6.5, 6.75, 9.25, 8.8, 8.8, 8.2, 8.9, 8.35, 7.5,
     6.75, 7.1, 7.5, 10.25, 8.25, 7.5, 5.1, 5.3, 5.2, 5, 7.25, 5.35, 6.1, 6, 6.6, 6.9, 7.25, 9.5, 6.5, 7.25, 7.25,
     7.8, 8.1, 8.2, 8.6, 9.4, 9.25, 8.8, 9.5, 9.25, 9.8, 10.25, 10.5, 10.75, 11.75, 9.8, 10.2, 11.25, 12.75, 14.1, 15.2, 12.5, 11.75,
     10.5, 9.75, 9.25, 8.25, 9, 9.25, 10, 10.4, 11.5, 12, 14.25, 14.25, 11.4])

df = pd.DataFrame({'x':x, 'y':y})
df.plot('x', 'y', kind='scatter')
print(len(x))
print(len(y))
#
#popt, pcov = curve_fit(F, x, y)
#plt.plot(x, y, 'r.')
#plt.plot(x, F(x,*popt), 'g-')


#def func(x, a, b, c):
#    return a * np.exp(-b * x) + c

#x = np.linspace(0,4,50)
#y = func(x, 2.5, 1.3, 0.5)
#yn = y + 0.2*np.random.normal(size=len(x))

#popt, pcov = curve_fit(func, x, y)
#plt.figure()
#plt.plot(x, y, 'ko', label="Original Noised Data")
#plt.plot(x, func(x, *popt), 'r-', label="Fitted Curve")
#plt.legend()


#ax = sns.regplot(x="x", y="y", data=df, scatter_kws={"s": 80}, order=2, ci=None, truncate=True)
ax = sns.regplot(x="x", y="y", data=df, x_estimator=np.mean, logx=True, truncate=False, scatter_kws={"color": "green"})
#model = np.polyfit(x,y,2)
#predictor = np.poly1d(model)

#xs = np.linspace(5000,5000, 100)

#plt.plot(x, y, '.') #, xs, predictor(xs), '-')
#plt.show()



#x = np.array([1,2,3,4,5,7,9,10,11])
#y = np.array([1,4,9,16,25,49,81,100,121])
#
#z = np.polyfit(x,y,2)
#print(z)
#
#p = np.poly1d(z)
#print(p(6))
#
#xp = np.linspace(1, 20, 20)
#
#plt.plot(x, y, '.', xp, p(xp), '-')
plt.show()



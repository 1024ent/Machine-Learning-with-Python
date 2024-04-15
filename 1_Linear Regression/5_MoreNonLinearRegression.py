'''
Author          : Loo Hui Kie
Contributors    : -
Title           : More Non-linear Regression example
Date Released   : 15/4/2024
'''
# Exploring Non-Linear Regression Functions with examples
'''  Import library  '''
import matplotlib.pyplot as plt
import numpy as np

''' Linear Regression '''
x = np.arange(-5.0, 5.0, 0.1)
y = 2 * x + 3
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata, 'bo')
plt.plot(x, y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

''' Cubic '''
y = 1*(x**3) + 1*(x**2) + 1*x + 3
y_noise = 20 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata,  'bo')
plt.plot(x,y, 'r') 
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

''' Quadratic '''
y = np.power(x, 2)
y_noise = 2 * np.random.normal(size=x.size)
ydata = y + y_noise
plt.plot(x, ydata, 'bo')
plt.plot(x, y, 'r')
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

''' Exponential '''
X = np.arange(0.1, 5.0, 0.1)
Y = np.exp(X)
plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

''' Logarithmic '''
X = np.arange(0.1, 5.0, 0.1)
Y = np.log(X)
plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

''' Sigmoidal/Logistic '''
X = np.arange(-5.0, 5.0, 0.1)
Y = 1 - 4 / (1 + np.power(3, X - 2))
plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

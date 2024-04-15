'''
Author          : Loo Hui Kie
Contributors    : -
Title           : Non-linear Regression
Date Released   : 15/4/2024
'''
''' Import library'''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

''' Read the data '''
df = pd.read_csv("china_gdp.csv")
print(df.head(10))

''' Exploring the data '''
# Plotting the Dataset
plt.figure(figsize=(8, 5))
x_data, y_data = df['Year'].values, df['Value'].values
plt.plot(x_data, y_data, 'ro')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()

''' Choosing model '''
# Logistic
X = np.arange(-5.0, 5.0, 0.1)
Y = 1.0 / (1.0 + np.exp(-X))
plt.plot(X, Y)
plt.ylabel('Dependent Variable')
plt.xlabel('Independent Variable')
plt.show()

''' Building Model '''
def sigmoid(x, Beta_1, Beta_2):
    y = 1 / (1 + np.exp(-Beta_1*(x-Beta_2)))
    return y

beta_1 = 0.10
beta_2 = 1990.0

# Logistic function
Y_pred = sigmoid(x_data, beta_1, beta_2)

# Plot initial prediction against datapoints
plt.plot(x_data, Y_pred * 15000000000000.)
plt.plot(x_data, y_data, 'ro')
plt.show()

# Lets normalize our data
xdata = x_data/max(x_data)
ydata = y_data/max(y_data)

# Use curve_fit fit sigmoid function to our data
from scipy.optimize import curve_fit
popt, pcov = curve_fit(sigmoid, xdata, ydata)
# Print final parameter
print(" beta_1 = %f, beta_2 = %f" % (popt[0], popt[1]))

# Plot regression model
x = np.linspace(1960, 2015, 55)
x = x/max(x)
plt.figure(figsize=(8, 5))
y = sigmoid(x, *popt)
plt.plot(xdata, ydata, 'ro', label='data')  # Corrected 'lable' to 'label'
plt.plot(x, y, linewidth=3.0, label='fit')  # Corrected 'lable' to 'label'
plt.legend(loc='best')
plt.ylabel('GDP')
plt.xlabel('Year')
plt.show()


'''  Practice  '''
# split data into train/test
msk = np.random.rand(len(df)) < 0.8
train_x = xdata[msk]
test_x = xdata[~msk]
train_y = ydata[msk]
test_y = ydata[~msk]

# build the model using train set
popt, pcov = curve_fit(sigmoid, train_x, train_y)

# predict using test set
y_hat = sigmoid(test_x, *popt)

# evaluation
print("Mean absolute error: %.2f" % np.mean(np.absolute(y_hat - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((y_hat - test_y) ** 2))
from sklearn.metrics import r2_score
print("R2-score: %.2f" % r2_score(test_y,y_hat) )
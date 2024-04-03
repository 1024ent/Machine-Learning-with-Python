'''  Import libraries  '''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

'''     Reading the data    '''
df = pd.read_csv("FuelConsumptionCo2.csv")
# # Print the 1st few row of the data
# print(df.head(10))

'''     Data exploration    '''
# summarize the data
print(df.describe())

# select features that we want to use for regression
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

# plot emission values with respect to Engine size:
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

'''     Creating train and test dataset     '''
# randomly select 80% of the data to train and 20% to test
msk     = np.random.rand(len(df)) < 0.8
train   = cdf[msk]
test    = cdf[~msk]

# train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

'''     Multiple Regression Model   '''
# in this case there are multiple linear regression model with 3 parameters, Sciki-learn uses OLS(Ordinary Least Squares) method to solve this issue
from sklearn import linear_model
regr    = linear_model.LinearRegression()
x_train = np.asanyarray(train[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_train = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(x_train, y_train)
# The coefficients
print('Coefficients             :', regr.coef_)

''' Prediction '''
x_test  = np.asanyarray(test[['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB']])
y_test  = np.asanyarray(test[['CO2EMISSIONS']])
y_hat   = regr.predict(x_test)
print("Residual sum of squares  : %.2f" % np.mean((y_hat - y_test) ** 2))
# Explained variance score: 1 is perfect prediction
print('Variance score           : %.2f' % regr.score(x_test, y_test))


'''     Practice    '''
# Use FUELCONSUMPTION_CITY and FUELCONSUMPTION_HWY instead of FUELCONSUMPTION_COMB
regrn = linear_model.LinearRegression()
# train
xn_train = np.asanyarray(train[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
regrn.fit(xn_train, y_train)
#  The coefficients
print('Coefficients             :', regrn.coef_)

# Evaluate/Predict
xn_test  = np.asanyarray(test[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_CITY','FUELCONSUMPTION_HWY']])
yn_hat   = regrn.predict(xn_test)
print("Risidual sum error       : %.2f"% np.mean((yn_hat - y_test) ** 2))
print('Variance score           : %.2f'% regrn.score(xn_test, y_test))
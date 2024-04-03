''' import necessary library '''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

'''     Reading the data    '''
df = pd.read_csv("FuelConsumptionCo2.csv")

# # Print the first few rows of the DataFrame
# print(df.head(10))

'''     Data Exploration   '''
# Summarize the data
print(df.describe())

# Select some features to explore more
cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
print(cdf.head(9))

# Plot each of the features
viz = cdf[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
viz.hist()
plt.show()

# Plot these features againt Emission to see how linear their relationship is:
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("FUELCONSUMPTION_COMB")
plt.ylabel("CO2EMISSIONS")
plt.show()

plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color="blue")
plt.xlabel("CYLINDERS")
plt.ylabel("CO2EMISSIONS")
plt.show()

'''  Creating train and test dataset  '''
# randomly select 80% of the data to train and 20% to test
msk     = np.random.rand(len(df)) < 0.8
train   = cdf[msk]
test    = cdf[~msk]

'''  Simple Regresiion Model  '''
# Train data distribution
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()

'''  Modeling  '''
# Using the sklearn package to model data
from sklearn import linear_model
regr    = linear_model.LinearRegression()
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])
regr.fit(train_x, train_y)
# The coefficients
print('Coefficients                 :', regr.coef_)
print('Intercept                    :', regr.intercept_)

# Plot output
plt.scatter(train.ENGINESIZE ,train.CO2EMISSIONS, color='blue')
plt.plot(train_x, regr.coef_[0][0]*train_x + regr.intercept_[0], '-r')
plt.xlabel("ENGINESIZE")
plt.ylabel("Emission")
plt.show()

'''  Evaluation  '''
from sklearn.metrics import r2_score

test_x  = np.asanyarray(test[['ENGINESIZE']])
test_y  = np.asanyarray(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)

print("Mean absolute error          : %.2f" % np.mean(np.absolute(test_y_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_y_ - test_y) ** 2))
print("R2-score                     : %.2f" % r2_score(test_y , test_y_) )



'''  Train using FUELCONSUMPTION_COMB  '''
regr_n  = linear_model.LinearRegression()
train_xn= np.asanyarray(train[['FUELCONSUMPTION_COMB']])
regr_n.fit(train_xn, train_y)
# The coefficients
print('Coefficients                 :', regr_n.coef_)
print('Intercept                    :', regr_n.intercept_)

# Plot output
plt.scatter(train.FUELCONSUMPTION_COMB ,train.CO2EMISSIONS, color='blue')
plt.plot(train_xn, regr_n.coef_[0][0]*train_xn + regr_n.intercept_[0], '-r')
plt.xlabel("FUEL CONSUMPTION")
plt.ylabel("Emission")
plt.show()

'''  Evaluation  '''
test_xn  = np.asanyarray(test[['FUELCONSUMPTION_COMB']])
test_yn_ = regr_n.predict(test_xn)

print("Mean absolute error          : %.2f" % np.mean(np.absolute(test_yn_ - test_y)))
print("Residual sum of squares (MSE): %.2f" % np.mean((test_yn_ - test_y) ** 2))
print("R2-score                     : %.2f" % r2_score(test_y , test_yn_))

# Result you get for comparing MAE of training using ENGINESIZE and FUELCONSUMPTION_COMB
# The MAE is much worse when we train using ENGINESIZE than FUELCONSUMPTION_COMB
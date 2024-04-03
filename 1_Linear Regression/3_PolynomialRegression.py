'''     Import Libraries    '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 

'''     Read the data   '''
df = pd.read_csv("FuelConsumptionCo2.csv")
# print(df.head(9))

'''     Data Exploration  '''
# Summarize data
print(df.describe())

cdf = df[['ENGINESIZE','CYLINDERS','FUELCONSUMPTION_COMB','CO2EMISSIONS']]
# print(cdf.head(9))

# Plot Emission values with respect to Engine Size:
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='blue')
plt.xlabel("Engine size")
plt.ylabel("Emissions")
plt.show()

'''     Creating train and test dataset     '''
msk     = np.random.rand(len(df)) < 0.8
train   = cdf[msk]
test    = cdf[~msk]

'''     Polynomial regression   '''
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
train_x = np.asanyarray(train[['ENGINESIZE']])
train_y = np.asanyarray(train[['CO2EMISSIONS']])

test_x = np.asanyarray(test[['ENGINESIZE']])
test_y = np.asanyarray(test[['CO2EMISSIONS']])

poly = PolynomialFeatures(degree=2)
train_x_poly = poly.fit_transform(train_x)
print(train_x_poly)

# Use linear regression to deal with linear regression problem
clf     = linear_model.LinearRegression()
clf.fit(train_x_poly, train_y)
# The coefficient
print('(Degree 2)Coefficients   :', clf.coef_)
print('(Degree 2)Intercept      :', clf.intercept_)

# Plot the trained result
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XX = np.arange(0.0, 10.0, 0.1)
yy = clf.intercept_[0]+ clf.coef_[0][1]*XX+ clf.coef_[0][2]*np.power(XX, 2)
plt.plot(XX, yy, '-r')
plt.xlabel("Engine Size")
plt.ylabel("Emission")
plt.show()

'''     Evaluation      '''
from sklearn.metrics import r2_score

test_x_poly = poly.transform(test_x)
test_y_     = clf.predict(test_x_poly)

print("(Degree 2)Mean absolute error    : %.2f"% np.mean(np.absolute(test_y_ - test_y)))
print("(Degree 2)Risidual sum error     : %.2f"% np.mean((test_y_ - test_y)** 2))
print("(Degree 2)R2-score               : %.2f"% r2_score(test_y, test_y_))

'''     Try degree 3 (cubic)    '''
poly3 = PolynomialFeatures(degree=3)
train_x_poly3 = poly3.fit_transform(train_x)
print(train_x_poly3)

clf3     = linear_model.LinearRegression()
clf3.fit(train_x_poly3, train_y)
# The coefficient
print('(Degree 3)Coeffiecients          :', clf3.coef_)
print('(Degree 3)Intecept               :', clf3.intercept_)

# Plot the trained result
plt.scatter(train.ENGINESIZE, train.CO2EMISSIONS, color='blue')
XXn = np.arange(0.0, 10.0, 0.1)
yyn = clf3.intercept_[0]+ clf3.coef_[0][1]*XXn + clf3.coef_[0][2]*np.power(XXn, 2) + clf3.coef_[0][3]*np.power(XXn, 3)
plt.plot(XXn, yyn, '-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")
plt.show()

test_x_poly3    = poly3.transform(test_x)
test_y_3        = clf3.predict(test_x_poly3)
print("(Degree 3)Mean absolute error    : %.2f"% np.mean(np.absolute(test_y_3 - test_y)))
print("(Degree 3)Risidual sum error     : %.2f"% np.mean(np.absolute((test_y_3 - test_y)**2)))
print("(Degree 3)R2-score               : %.2f"% r2_score(test_y, test_y_3))
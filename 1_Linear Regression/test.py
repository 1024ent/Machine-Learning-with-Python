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

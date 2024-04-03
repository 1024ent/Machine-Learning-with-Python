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
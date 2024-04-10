'''  Import libraries  '''
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split


'''  Read Data  '''
data = pd.read_csv("real_estate_data.csv")
# print(data.head(10))
# print(data.shape)
# print(data.isna().sum) # Check wheter there's missing value in data

'''  Data Preprocessing  '''
data.dropna(inplace=True) # Drop missing value so that we ahve enough data for our dataset
# print(data.isna().sum) # Now we can see our data has no missing value

# split the dataset into our features and what we are predicting (target)
X = data.drop(columns=["MEDV"])
Y = data["MEDV"]
# print(X.head(10))
# print(Y.head(10))

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=.2 , random_state= 1)

'''  Create Regression Tree  '''
regression_tree = DecisionTreeRegressor(criterion = "squared_error")

'''  Training  '''
regression_tree.fit(X_train, Y_train)

'''  Evaluation  '''
print(regression_tree.score(X_test, Y_test))
prediction = regression_tree.predict(X_test)
print("$",(prediction - Y_test).abs().mean()*1000)

'''  Practice Use Mae  '''
mae_regression_tree = DecisionTreeRegressor(criterion= "absolute_error")
mae_regression_tree.fit(X_train, Y_train)
print(mae_regression_tree.score(X_test, Y_test))
mae_prediction = mae_regression_tree.predict(X_test)
print("$",(mae_prediction - Y_test).abs().mean()*1000)


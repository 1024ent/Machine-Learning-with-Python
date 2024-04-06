# KNN (K-Nearest Neighbours) - This code implements the K-Nearest Neighbors algorithm

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

# Load the data from the CSV file
df = pd.read_csv('teleCust1000t.csv')

# Print a sample of the first 10 rows to see the data (optional)
# print(df.head(10))

'''  Data Visualization and Analysis  '''
# Explore the data by getting the number of customers in each category
value_counts = df['custcat'].value_counts().sort_index()

# Print the counts with the customer categories
for value, count in value_counts.items():
    print(f"Customer Category {value}: {count} people")  # f-string for formatted printing

# Plot a histogram to see the distribution of income
df.hist(column='income', bins=50)
plt.xlabel('Income')
plt.ylabel('Number of people')
plt.title('Distribution of income in sample data')
plt.show()

# Get the feature columns we will use for modeling (excluding non-numerical ones)
feature_columns = ['region', 'tenure','age', 'marital', 'address', 'income', 'ed', 'employ','retire', 'gender', 'reside']
X = df[feature_columns].values.astype(float)  # Convert to float for some algorithms in scikit-learn

# Get the target variable (customer category)
y = df['custcat'].values

'''  Normalize Data  '''
# Standardize the features to have a mean of 0 and standard deviation of 1 (often recommended)
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

'''  Train Test Split '''
# Split the data into training and testing sets. 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)

print('Training set:', X_train.shape,  y_train.shape)  # Show the shapes (number of rows, columns)
print('Test set:', X_test.shape,  y_test.shape)

'''  Classification - KNN Algorithm '''
# Create a KNN classifier object. Here, k (number of neighbors) is set to 4 initially
k = 4
neigh = KNeighborsClassifier(n_neighbors=k).fit(X_train, y_train)

# Use the trained model to predict customer categories for the test set
yhat = neigh.predict(X_test)
print(yhat[0:5])  # Print the first 5 predicted categories

# Evaluate the performance of the model on the test set - Accuracy
print("Test set Accuracy:", metrics.accuracy_score(y_test, yhat))

'''  Practice with different K values '''
# Try training the model with a different k value (k=6)
kn = 6
neigh6 = KNeighborsClassifier(n_neighbors=kn).fit(X_train, y_train)
yhat6 = neigh6.predict(X_test)
print(yhat6[0:5])  # Print the first 5 predicted categories with k=6

# Evaluate the performance with k=6
print("Test set Accuracy with k=6:", metrics.accuracy_score(y_test, yhat6))

'''  Choosing the optimal K value  '''
# Explore different k values (from 1 to Ks) and record their accuracy
Ks = 50
mean_acc = np.zeros((Ks - 1))
std_acc = np.zeros((Ks - 1))

for n in range(1,Ks):
    
    #Train Model and Predict  
    neigh = KNeighborsClassifier(n_neighbors = n).fit(X_train,y_train)
    yhat=neigh.predict(X_test)
    mean_acc[n-1] = metrics.accuracy_score(y_test, yhat)

    
    std_acc[n-1]=np.std(yhat==y_test)/np.sqrt(yhat.shape[0])

print(mean_acc)

# Plot the average accuracy vs number of neighbors (k)
plt.plot(range(1,Ks),mean_acc,'g')
plt.fill_between(range(1,Ks),mean_acc - 1 * std_acc,mean_acc + 1 * std_acc, alpha=0.10)
plt.fill_between(range(1,Ks),mean_acc - 3 * std_acc,mean_acc + 3 * std_acc, alpha=0.10,color="green")
plt.legend(('Accuracy ', '+/- 1xstd','+/- 3xstd'))
plt.ylabel('Accuracy ')
plt.xlabel('Number of Neighbors (K)')
plt.tight_layout()
plt.show()

print( "The best accuracy was with", mean_acc.max(), "with k=", mean_acc.argmax()+1) 
'''
Author          : Loo Hui Kie
Contributors    : -
Title           : Decission Tree
Date Released   : 15/4/2024
'''
'''  Import Libraries  '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

'''  Read Data  '''
data = pd.read_csv("drug200.csv", delimiter=",")
# print(data[0:5])
# print(data.shape) # Find the size of data

'''  Data Pre-Processing  '''
X = data[['Age', 'Sex', 'BP', 'Cholesterol','Na_to_K']].values
# print(X[0:5])

from sklearn import preprocessing
le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])
# print(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit(['LOW', 'NORMAL', 'HIGH'])
X[:,2] = le_BP.transform(X[:,2])
# print(X[:,2])

le_chol = preprocessing.LabelEncoder()
le_chol.fit(['NORMAL', 'HIGH'])
X[:,3] = le_chol.transform(X[:,3])
# print(X[:,3])
# print(X[0:5])

y = data[['Drug']]
# print(y[0:5])

'''  Setting Up Decission Tree  '''
from sklearn.model_selection import train_test_split
X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size= 0.3, random_state=3)
print('Shape from X_trainset{}'.format(X_trainset.shape),'&','Shape from y_trainset{}'.format(y_trainset.shape))
print('Shape from X_testset{}'.format(X_testset.shape),'&','Shape from y_testset{}'.format(y_testset.shape))

'''  Modeling  '''
drugtree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
# print(drugtree) # it shows the default parameters

drugtree.fit(X_trainset, y_trainset)

'''  Prediction  '''
predtree = drugtree.predict(X_testset)
print(predtree[0:5])
print(y_testset[0:5])

'''  Evaluation  '''
from sklearn import metrics
print("DecissionTree's Accuracy :", metrics.accuracy_score(y_testset, predtree))

'''  Visualisation  '''
import sklearn.tree as tree
tree.plot_tree(drugtree)
plt.show()
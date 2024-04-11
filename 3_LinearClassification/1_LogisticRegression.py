''' Import library  '''
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import scipy.optimize as opt

''' Data Analysis  '''
churn_df = pd.read_csv('ChurnData.csv')

'''  Data Preprocess and Selection  '''
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
print(churn_df.head())

print("There are " + str(len(churn_df)) + " observations in the datasets")
print("There are " + str(len(churn_df.columns)) + " variables in the datasets")
print("churn_df.shape= ", churn_df.shape)

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless']])
print(X[0:5])

# ravale(), reshape the y variable into a 1D array
y = np.asarray(churn_df[['churn']]).ravel()
print(y[0:5])

X = preprocessing.StandardScaler().fit(X).transform(X)
print(X[0:5])

''' Train, Test and Split  '''
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

'''  Modeling (Logistic Regression with Scikit-learn)  '''
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# solver = ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
print(LR)

# predict using our test set
yhat = LR.predict(X_test)

# predict_proba returns estimates for all classes, ordered by the label of classes
yhat_prob = LR.predict_proba(X_test)
print(yhat_prob)

'''  Evaluation  '''
# jaccard index
from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat, pos_label=0))

# confusion matrix
from sklearn.metrics import classification_report, confusion_matrix
import itertools

def plot_confusion_matrix(cm, classes,
                        normalize=False,
                        title='Confusion matrix',
                        cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

print(confusion_matrix(y_test, yhat, labels=[1,0]))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1', 'churn=0'], normalize=False, title='Confusion Matrix')

print(classification_report(y_test, yhat))

# Log loss
from sklearn.metrics import log_loss
print(log_loss(y_test, yhat_prob))

''''  Practice  '''
# build Logistic Regression model again for the same dataset, but this time, use different __solver__ and __regularization__ values? What is new __logLoss__ value? 
LR_sag = LogisticRegression(C=0.01, solver='sag').fit(X_train, y_train)
yhat_prob_sag = LR_sag.predict_proba(X_test)
print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob_sag))
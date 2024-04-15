'''
Author          : Loo Hui Kie
Contributors    : -
Title           : Support Vector Machine
Date Released   : 15/4/2024
'''
'''  Import library  '''
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

'''  Data Analysis  '''
cell_df = pd.read_csv('cell_samples.csv')
# print(cell_df.head(10))
ax = cell_df[cell_df['Class']==4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class']==2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
plt.show()

'''  Data pre-processing and selection  '''
# look at columns data types
print(cell_df.dtypes)

# BareNuc column includes some values that are not numerical. We can drop those rows
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
print(cell_df.dtypes)

feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
print(X[0:5])

cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])
print(y[0:5])

'''  Train/Test dataset  '''
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=4)
print("X_train.shape= ", X_train.shape, "X_test.shape= ", X_test.shape)
print("y_train,shape= ", y_train.shape, "y_test.shape= ", y_test.shape)

'''  Modeling (SVM with Scikit-learn)  '''
from sklearn import svm
clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)

yhat = clf.predict(X_test)
print(yhat[0:5])

'''  Evaluation  '''
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

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

print(classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['Benign(2)', 'Malignant(4)'], normalize=False, title="Confusion Matrix")

from sklearn.metrics import f1_score
print(f1_score(y_test, yhat, average='weighted'))

from sklearn.metrics import jaccard_score
print(jaccard_score(y_test, yhat, pos_label=2))

'''  Practice  '''
# Use linear kernel to build model
clf_linear = svm.SVC(kernel='linear')
clf_linear.fit(X_train, y_train)

yhat_linear = clf_linear.predict(X_test)

cnf_linear = confusion_matrix(y_test ,yhat_linear, labels=[2,4])

plot_confusion_matrix(cnf_linear, classes=['Benign(2)', 'Malignant(4)'], normalize=False, title="Confusion Matrix")

print(f1_score(y_test, yhat_linear, average='weighted'))
print(jaccard_score(y_test, yhat_linear, pos_label=2))
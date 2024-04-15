'''
Author          : Loo Hui Kie
Contributors    : -
Title           : Hierarchical Clustering 
Date Released   : 15/4/2024
'''
'''  Import library  '''
import numpy as np 
import pandas as pd
from scipy import ndimage 
from scipy.cluster import hierarchy 
from scipy.spatial import distance_matrix
from scipy.spatial.distance import squareform 
from matplotlib import pyplot as plt 
from sklearn import manifold, datasets 
from sklearn.cluster import AgglomerativeClustering 
from sklearn.datasets import make_blobs 

'''  Generating Random Data  '''
X1, y1 = make_blobs(n_samples=50, centers=[[4,4], [-2, -1], [1, 1], [10,4]], cluster_std=0.9)

plt.scatter(X1[:, 0], X1[:, 1], marker='o') 

'''  Agglomerative Clustering  '''
agglom = AgglomerativeClustering(n_clusters = 4, linkage = 'average')

agglom.fit(X1,y1)

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(6,4))

# These two lines of code are used to scale the data points down,
# Or else the data points will be scattered very far apart.

# Create a minimum and maximum range of X1.
x_min, x_max = np.min(X1, axis=0), np.max(X1, axis=0)

# Get the average distance for X1.
X1 = (X1 - x_min) / (x_max - x_min)

# This loop displays all of the datapoints.
for i in range(X1.shape[0]):
    # Replace the data points with their respective cluster value 
    # (ex. 0) and is color coded with a colormap (plt.cm.spectral)
    plt.text(X1[i, 0], X1[i, 1], str(y1[i]),
            color=plt.cm.nipy_spectral(agglom.labels_[i] / 10.),
            fontdict={'weight': 'bold', 'size': 9})
    
# Remove the x ticks, y ticks, x and y axis
plt.xticks([])
plt.yticks([])
#plt.axis('off')

# Display the plot of the original data before clustering
plt.scatter(X1[:, 0], X1[:, 1], marker='.')
# Display the plot
plt.show()

'''  Dendrogram Associated for the Agglomerative Hierarchical Clustering  '''
# Compute the distance matrix
dist_matrix = distance_matrix(X1, X1)

# Convert the distance matrix to condensed form
condensed_dist = squareform(dist_matrix)

Z = hierarchy.linkage(condensed_dist, 'complete')
dendro = hierarchy.dendrogram(Z)

'''  Practice  '''
Z = hierarchy.linkage(condensed_dist, 'average')
dendro = hierarchy.dendrogram(Z)


'''  Clustering on Vehicle dataset  '''
'''  Read data  '''
pdf = pd.read_csv('cars_clus.csv')
print ("Shape of dataset: ", pdf.shape)
print(pdf.head(5))

'''  Data Cleaning  '''
print ("Shape of dataset before cleaning: ", pdf.size)
pdf[[ 'sales', 'resale', 'type', 'price', 'engine_s',
    'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
    'mpg', 'lnsales']] = pdf[['sales', 'resale', 'type', 'price', 'engine_s',
    'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap',
    'mpg', 'lnsales']].apply(pd.to_numeric, errors='coerce')
pdf = pdf.dropna()
pdf = pdf.reset_index(drop=True)
print ("Shape of dataset after cleaning: ", pdf.size)
print(pdf.head(5))

'''  Feature selection  '''
featureset = pdf[['engine_s',  'horsepow', 'wheelbas', 'width', 'length', 'curb_wgt', 'fuel_cap', 'mpg']]

'''  Normalization  '''
from sklearn.preprocessing import MinMaxScaler
x = featureset.values #returns a numpy array
min_max_scaler = MinMaxScaler()
feature_mtx = min_max_scaler.fit_transform(x)
print(feature_mtx [0:5])

'''  Clustering using Scipy  '''
import scipy

leng = feature_mtx.shape[0]

# Compute the distance matrix
D = np.zeros([leng, leng])
for i in range(leng):
    for j in range(leng):
        D[i, j] = scipy.spatial.distance.euclidean(feature_mtx[i], feature_mtx[j])

# Convert the distance matrix to condensed form
condensed_dist = squareform(D)
print(condensed_dist)

import pylab
import scipy.cluster.hierarchy
Z = hierarchy.linkage(condensed_dist, 'complete')

from scipy.cluster.hierarchy import fcluster
max_d = 3
clusters = fcluster(Z, max_d, criterion='distance')
print(clusters)

from scipy.cluster.hierarchy import fcluster
k = 5
clusters = fcluster(Z, k, criterion='maxclust')
print(clusters)

fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

'''  Clustering using scikit-learn  '''
from sklearn.metrics.pairwise import euclidean_distances
import warnings
# Compute the distance matrix
dist_matrix = euclidean_distances(feature_mtx, feature_mtx)

# Check the shape of dist_matrix
print("Shape of dist_matrix:", dist_matrix.shape)

# Suppress the specific warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)  # Adjust the warning category as needed
    Z_using_dist_matrix = hierarchy.linkage(dist_matrix, 'complete')

# Now you can use Z_using_dist_matrix for further analysis or visualization

fig = pylab.figure(figsize=(18,50))
def llf(id):
    return '[%s %s %s]' % (pdf['manufact'][id], pdf['model'][id], int(float(pdf['type'][id])) )
    
dendro = hierarchy.dendrogram(Z_using_dist_matrix,  leaf_label_func=llf, leaf_rotation=0, leaf_font_size =12, orientation = 'right')

# Suppress the specific warning
with warnings.catch_warnings():
    warnings.simplefilter("ignore", category=UserWarning)  # Adjust the warning category as needed
    agglom = AgglomerativeClustering(n_clusters=6, linkage='complete')
    agglom.fit(dist_matrix)

# Get the cluster labels
labels = agglom.labels_

pdf['cluster_'] = agglom.labels_
print(pdf.head())

import matplotlib.cm as cm
n_clusters = max(agglom.labels_)+1
colors = cm.rainbow(np.linspace(0, 1, n_clusters))
cluster_labels = list(range(0, n_clusters))

# Create a figure of size 6 inches by 4 inches.
plt.figure(figsize=(16,14))

for color, label in zip(colors, cluster_labels):
    subset = pdf[pdf.cluster_ == label]
    for i in subset.index:
            plt.text(subset.horsepow[i], subset.mpg[i],str(subset['model'][i]), rotation=25) 
    plt.scatter(subset.horsepow, subset.mpg, s= subset.price*10, color=color, label='cluster'+str(label),alpha=0.5)
#    plt.scatter(subset.horsepow, subset.mpg)
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()

pdf.groupby(['cluster_','type'])['cluster_'].count()

agg_cars = pdf.groupby(['cluster_', 'type'])[['horsepow', 'engine_s', 'mpg', 'price']].mean()

print(agg_cars)

plt.figure(figsize=(16, 10))
for color, label in zip(colors, cluster_labels):
    subset = agg_cars.loc[(label,),]
    for i in range(len(subset)):  # Iterate over the range of the length of subset
        plt.text(subset.iloc[i, 0] + 5, subset.iloc[i, 2], 'type=' + str(int(subset.index[i])) + ', price=' + str(int(subset.iloc[i, 3])) + 'k')
    plt.scatter(subset.horsepow, subset.mpg, s=subset.price * 20, color=color, label='cluster' + str(label))
plt.legend()
plt.title('Clusters')
plt.xlabel('horsepow')
plt.ylabel('mpg')
plt.show()
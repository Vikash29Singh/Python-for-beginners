# -*- coding: utf-8 -*-
"""

https://www.kaggle.com/baiazid/pima-indians-diabetes-na-ve-bayes
Created on Mon Jan 27 12:46:25 2020

@author: vikash
"""
#import the libraries
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import pandas_profiling 
import matplotlib.pyplot as plt #Data Visualization 
import seaborn as sns  #Python library for Vidualization



#Import the dataset

dataset = pd.read_csv('Mall_Customers.csv')
dataset.info()
print(dataset.isnull().sum())



dataset.head(10) #Printing first 10 rows of the dataset

#total rows and colums in the dataset
dataset.shape

dataset.info() # there are no missing values as all the columns has 200 entries properly

#Missing values computation
dataset.isnull().sum()

### Feature sleection for the model
#Considering only 2 features (Annual income and Spending Score) and no Label available
X= dataset.iloc[:, [3,4]].values

#Building the Model
#KMeans Algorithm to decide the optimum cluster number 
from sklearn.cluster import KMeans
wcss=[]

#we always assume the max number of cluster would be 10
#you can judge the number of clusters by doing averaging
###Static code to get max no of clusters

for i in range(1,11):
    kmeans = KMeans(n_clusters= i, init='k-means++', random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

    #inertia_ is the formula used to segregate the data points into clusters
    
#Visualizing the ELBOW method to get the optimal value of K 
plt.plot(range(1,11), wcss)
plt.title('The Elbow Method')
plt.xlabel('no of clusters')
plt.ylabel('wcss')
plt.show()

#Model Build
kmeansmodel = KMeans(n_clusters= 5, init='k-means++', random_state=0)
y_kmeans= kmeansmodel.fit_predict(X)


#Visualizing all the clusters 

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Earning high')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Average high')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Spend high')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'earning less')
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Both less')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

print("Model Interpretation \n")
print("Earning high (Red Color) -> earning high but spending less \n")
print("Average high (Blue Colr) -> average in terms of earning and spending \n ")
print("Spend high (Green Color) -> earning high and also spending high [TARGET SET] \n")
print("earning less (cyan Color) -> earning less but spending more \n")
print("Both less (magenta Color) -> Earning less , spending less \n")

print("We can put Cluster 3 into some alerting system where email can be send to them on daily basis as these re easy to converse ###### \n")
print("wherein others we can set like once in a week or once in a month \n")



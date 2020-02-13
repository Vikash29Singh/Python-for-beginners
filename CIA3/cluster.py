# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 11:44:47 2020

@author: vikash
"""

import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

data=pd.read_csv("height_weight.csv")
print(data.head(20))
data=data.head(100)
data.drop("Index",axis=1,inplace=True)
print(data)
plt.scatter(data["Height(Inches)"],data["Weight(Pounds)"])
#plt.show()
X=data.to_numpy()
#print(l)
#print(type(l))
# BASE CODE

#X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
#print(X)
#print(X[:,1])
plt.title('Total Clusters')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.scatter(X[:,0], X[:,1])
plt.show()
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

print("Enter the number of clusters ")
clus=int(input())

kmeans = KMeans(n_clusters=clus, init='k-means++', max_iter=300, n_init=10, random_state=0)
pred_y = kmeans.fit_predict(X)
y_kmeans = kmeans.predict(X)


plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)

#plt.scatter(X[:,0], X[:,1])
#plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')

plt.title('Final Clusters')
plt.xlabel('Height')
plt.ylabel('Weight')
plt.show()

print("Centres")
print("Index - Height - Weight")
for x in range(len(centers)):
    print((x+1),"\t",round(centers[x][0],2),"-",round(centers[x][1],2))

print("Total Clusters = ",clus)

for i in range(clus):  
    l=[]
    for index,value in enumerate(y_kmeans):
        if value == i:
            l.append([X[index][0],X[index][1]])
    print("Cluster - ",i)
    #print(l)
print(y_kmeans)    

# -*- coding: utf-8 -*-
"""
@author: Rachel Sumner (34559248)
"""

import math
import matplotlib.pyplot as plt
import numpy as np
import pylab
import scipy.cluster as sc
import scipy.spatial.distance as sd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier, plot_tree
import statsmodels.api as sm
#from mpl_toolkits import mplot3d


"""
Preprocessing the Manchester weather data set
Normalization, standardisation, PCA
"""

# Standardisation

def standardise(data):
    #Similar to my solution to lab 3, but with global outlier removal
    #Standardizes the data by calculating their Z-score. Most will be within [-3,3]
    
    standardised_data = data.copy()  
    rows = data.shape[0]
    columns = data.shape[1]

    for j in range(columns):
        #goes through each feature and standardizes them within [-3,3] (j is the feature, i is the thing)
        mean = np.mean(data[:,j])
        sigma = np.std(data[:,j])
 
        for i in range(rows):
            standardised_data[i,j] = (data[i,j] - mean) / sigma
    
    # Remove global outliers (those outside [-3,3])
    no_outlier_data = []
    #counter = 0
    for i in range(rows):
        
        row = standardised_data[i]
        row = row[abs(row) <= 3] # Removes elements with standard deviation of 3 or more sigma
        if len(row) == 5:
            # Only adds non-outliers to the new array
            no_outlier_data.append(row)
        #else:
            #counter = counter + 1
            #print(counter)
    no_outlier_data = np.array(no_outlier_data)
    
    return no_outlier_data

# Normalization

def normalize(data):
    # Same as my solution to lab 3
    # Normalizes the data around [-1,1]
    normalised_data = data.copy()
    rows = data.shape[0]
    columns = data.shape[1]

    for j in range(columns):
        
        max_element = np.amax(data[:,j])
        min_element = np.amin(data[:,j])

        for i in range(rows):        
            normalised_data[i,j] = 2 * (data[i,j] - min_element) / (max_element - min_element) - 1
        
    return normalised_data


# Centralisation

def centralise(data):
    # Same as the solution to lab 4
    # Centralises data, i.e translates the distribution such that the mean is at 0
    centralised_data = data.copy()
    rows = data.shape[0]
    columns = data.shape[1]
    
    for j in range(columns):
        mean = np.amax(data[:,j])
        
        for i in range(rows):
            centralised_data[i,j] = data[i,j] - mean
            
    return centralised_data


# Load the data
    
def load_data(file):
    # From lab 2
    # Reads a csv or txt and then returns a numpy array, containing each data point and their values for the given features
    data_raw = [];
    data_file = open(file, "r")
    
    while True:
        the_line = data_file.readline()
        if len(the_line) == 0:
             break  
        read_data = the_line.split(",")
        for pos in range(len(read_data)):
            read_data[pos] = float(read_data[pos]);
        data_raw.append(read_data)
    
    data_file.close()
    
    return np.array(data_raw)
 
       
data = load_data("SCC403ResitCWClimateData.csv")
standardised_data = np.array(standardise(data))
normalized_data = np.array(normalize(standardised_data))
centralised_data = np.array(centralise(normalized_data))

# Principal Component analysis

pca = PCA(n_components = 5) # There's 5 features, so the highest amount of components is 5
pca.fit(centralised_data)
Coeff = pca.components_
print("PCA coefficients:",Coeff) # Gives us the weights for each feature in each component

transformed_data = pca.transform(centralised_data)
variance = [0,0,0,0,0]
for i in range(len(variance)):
    # Calculate the total variance accounted for in each n components, for making a scree plot
    if i > 0:
        variance[i] = variance[i-1] + pca.explained_variance_ratio_[i]
    else:
        variance[i] = pca.explained_variance_ratio_[0]
        
plt.figure(figsize=(6,4))
plt.bar([1,2,3,4,5], pca.explained_variance_ratio_, tick_label=[1,2,3,4,5])
#x_values = np.arange(4)+1
#plt.plot(x_values, variance, 'ro-')
plt.xlabel("Principal component (no.)")
plt.ylabel("Variance Explained (%)")
plt.savefig("PCAvariance.pdf")
plt.show()

plt.figure(figsize=(6,4))
plt.plot(transformed_data[:,0], transformed_data[:,1], ".")
plt.xlabel("First principal component")
plt.ylabel("Second principal component")
plt.savefig("PCAfirsttwo.pdf")
plt.show()

"""
# first 3 PCA components
fig = plt.figure()
ax = plt.axes(projection='3d')
#ax.xlabel("First principal component")
#ax.ylabel("Second principal component")
#ax.zlabel("Third principal component")
ax.scatter3D(transformed_data[:,0], transformed_data[:,1], transformed_data[:,2], ".")
plt.savefig("PCAfirstthree.pdf")
plt.show()
"""

# QQ plot comparing our distribution with the normal distribution
sm.qqplot(transformed_data[:,0], line="45")
sm.qqplot(transformed_data[:,1], line="45")
pylab.show()



"""
Clustering
"""

# Hierarchical Clustering

def dist(x1, x2):
    # Taken from Lab 5 solutions
    # Calculates the Euclidean distance between two points
    total = 0

    for i in range(len(x1)):
        total = total + pow((x1[i] - x2[i]),2)

    return math.sqrt(total)

def distance(data):
    # Taken from Lab 5 solutions
    # Constructs the Proximity matrix used in Hierarchical Clustering
    rows = data.shape[0]
    columns = data.shape[1]
    
    distance_matrix = np.zeros((rows,rows))

    for i in range(rows):
        for j in range(rows):

            total = 0

            for c in range(columns):

                total += pow((data[i,c] - data[j,c]),2)

            distance_matrix[i,j] = math.sqrt(total)

    return distance_matrix

# We now construct the proximity matrix
"""
distance_data = distance(transformed_data)

condensed_distance = sd.squareform(distance_data)

Z = sc.hierarchy.linkage(condensed_distance) # Linkage information
#print(Z)

# Makes a dendogram showing only the last 20 levels, any more would be unwieldy
plt.figure(figsize=(6,4))

sc.hierarchy.dendrogram(Z, truncate_mode="level",p=20)

plt.savefig("dendrogramLevel20.pdf")

plt.close()
"""
# k-means clustering, we use the same dist() function as above, based on Lab 5

centroids = [0,0,0,0,0,0,0,0,0,0]
distortion = [0,0,0,0,0,0,0,0,0,0]

k = 10 
for i in range(0,k):
    """
    We generate random centroids and find the sum of squared distances so that 
    we may make a scree plot to find the best k
    """
    centroids[i], distortion[i] = sc.vq.kmeans(transformed_data, i+2)
x_values = np.arange(k) + 2
plt.figure(figsize=(6,4))

plt.plot(x_values, distortion, "bo-")
plt.title("Scree Test")
plt.xlabel("Number of clusters")
plt.ylabel("Sum of squared distances")
plt.savefig("kmeansscree.pdf")
plt.show()
plt.close()

# Now that we've found the best amount of clusters (6), we can make a plot showing them

centroids, distortion = sc.vq.kmeans(transformed_data, 6)
# This will contain all of our clusters
groups = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]

def sum_dist(m1 , m2):
    # Lab 5 solutions
    sumTotal = 0

    for pos in range(len(m1)):
        sumTotal = sumTotal + dist(m1[pos ,:],m2[pos ,:])

    return sumTotal

clusters = np.zeros(len(transformed_data)) # Stores what cluster everything's in

dist_centroids = float("inf")
threshold = 0.00005
while dist_centroids > threshold:
    """
    Modified from lab 5 solutions
    Simple k-means algorithm:
    We start with a set of random centroids, calculate the distance to them for each point,
    assign each point to the closest centroid's cluster, then find new centroids based on the new clusters.
    We repeat until there is no change in the centroids (i.e the clusters are stable)
    """
    groups = [np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
    centroids_old = centroids.copy()
    for j in range(len(transformed_data)):
        # Finds the nearest centroid
        dist_array = []
        for i in range(6):
            # Calculate distance from the point to every centroid
            dist_array.append(dist(transformed_data[j], centroids[i]))
            
        min_dist_index = dist_array.index(min(dist_array)) # Finds the index of the minimum array
        #print(dist_array, min_dist)

        if len(groups[min_dist_index]) == 0:
            groups[min_dist_index] = transformed_data[j]
            
        else:
            groups[min_dist_index] = np.vstack((groups[min_dist_index],transformed_data[j]))
        clusters[j] = min_dist_index
            
    # Now we have to find the mean for each current cluster so we can make
    # a better set of clusters next time
    for i in range(6):
        centroids[i] = np.mean(groups[i], axis=0)
    
    dist_centroids = sum_dist(centroids, centroids_old)
    print(dist_centroids)
print("clustering done")   
print(clusters)  
    
   
"""


while dist_centroids > threshold:
    # Modified from lab 5 solutions
    for i in range(len(transformed_data)):
        # Finds the nearest centroid
        dist_array = []
        for j in range(6): #k=6
            # Calculate distance from the point to every centroid
            dist_array.append(dist(transformed_data[i], centroids[j]))
            
        min_dist_index = dist_array.index(min(dist_array))
        clusters[i] = min_dist_index
    
    centroids_old = centroids.copy()
    
    for i in range(6): #k=6
        points = np.array([])
    
        for j in range(len(transformed_data)):
            if (clusters[j] == i):
                if (len(points) == 0):
                    points = transformed_data[j,:].copy()
                else:
                    points = np.vstack((points, transformed_data[j,:]))
        
        centroids[i] = np.mean(points, axis=0)
    
    dist_centroids = sum_dist(centroids, centroids_old)
"""
    
plt.figure(figsize=(6,4))

plt.plot(groups[0][:,0],groups[0][:,1],'r.')
plt.plot(groups[1][:,0],groups[1][:,1],'g.')
plt.plot(groups[2][:,0],groups[2][:,1],'b.')
plt.plot(groups[3][:,0],groups[3][:,1],'c.')
plt.plot(groups[4][:,0],groups[4][:,1],'m.')
plt.plot(groups[5][:,0],groups[5][:,1],'y.')

plt.plot(centroids[0,0],centroids[0,1],'rx')
plt.plot(centroids[1,0],centroids[1,1],'gx')
plt.plot(centroids[2,0],centroids[2,1],'bx')
plt.plot(centroids[3,0],centroids[3,1],'cx')
plt.plot(centroids[4,0],centroids[4,1],'mx')
plt.plot(centroids[5,0],centroids[5,1],'yx')

plt.xlabel("PCA First component")
plt.ylabel("PCA Second component")
plt.savefig("kmeansClassified.pdf")

plt.close()

print("centroids:",centroids)
"""
This is so we can see what each centroid can be labeled as, and if a point is in its cluster we label it the same
E.g. centroids[0] has 0.17399213  0.04638744  0.64326765 -0.29764445 -0.01468758
So we might classify it as above average temperature, still, easterly wind, low precipitation, average humidity
We can reduce this further to average temperature, still, dry
The main problem is the centroids ordering changes each time.
For the first time they're all as follows, and we will base our labels on this,
taking these as representative of their clusters
 [[ 0.17399213  0.04638744  0.64326765 -0.29764445 -0.01468758]
 [-0.80460864  0.75891421  0.25140354  0.17661368 -0.06939887]
 [ 0.83326971 -0.16543413  0.20903415  0.27799846  0.01436642]
 [-0.68001377 -0.31359424  0.03064589  0.00114577  0.00286512]
 [ 0.27210282 -0.2489207  -0.37574992 -0.07288547 -0.03876675]
 [ 0.2713497   0.71412959 -0.38907763 -0.0327368   0.09621123]]
 Our first class is: average temperature, average wind
 Our second class: cold, very windy, and wet
 Third: hot, average wind, and wet
 Fourth: cold and low wind
 Fifth: slightly hot, low wind
 Sixth: slightly hot, very windy
 I felt it best to exclude most that are average
 This isnt the best method but picking reprensative samples is the simplest way!
 The labelling isnt any more arbitrary than any decision method.
 Now I just have to get these labels onto the points somehow.
"""



"""
Classification
"""
labels = ["average temperature, average wind", "cold, very windy, and wet",
         "hot, average wind, and wet", "cold and low wind", "slightly hot and low wind",
         "slightly hot and very windy"]
labelled_data = np.zeros(len(transformed_data))

class_centroids = [[ 0.17399213,  0.04638744,  0.64326765, -0.29764445, -0.01468758],
 [-0.80460864,  0.75891421,  0.25140354,  0.17661368, -0.06939887],
 [ 0.83326971, -0.16543413,  0.20903415,  0.27799846,  0.01436642],
 [-0.68001377, -0.31359424,  0.03064589,  0.00114577,  0.00286512],
 [ 0.27210282, -0.2489207,  -0.37574992, -0.07288547, -0.03876675],
 [ 0.2713497,   0.71412959, -0.38907763, -0.0327368,  0.09621123]]
# The centroids we're using for classification, as described above
# First step is to get these the same as the actual centroids,
# they'll differ by very small amounts so we only need to check that
new_groups =  [np.array([]), np.array([]), np.array([]), np.array([]), np.array([]), np.array([])]
for j in range(len(class_centroids)):
    for i in range(len(centroids)):
        c_distance = dist(class_centroids[j],centroids[i])
        if c_distance < threshold:
            class_centroids[j] = centroids[i] # Syncs up the label with the centroid
            new_groups[j] = groups[i] # Permute the groups
            #for k in range(len(clusters)):
               # if labelled_data[k] == 0 and clusters[k] == i:
                   # labelled_data[k] = i
        break
            

for i in range(len(transformed_data)):
    for g in new_groups:
        if transformed_data[i] in g:
            labelled_data[i] = g.index
print(labelled_data)
print("class centroids:",class_centroids)            

print(labelled_data)
# Barely modified from lab 7
n_classes = 6 # Just cos we had 6 clusters
plot_colors = "rgbcmy"
plot_step = 0.02

# Load data

for pairidx, pair in enumerate([[0, 1], [0, 2], [0, 3], #[0, 4], [0, 5], [0, 6],
                                [1, 2], [1, 3], [2, 3], #[1, 4], [1, 5], [2, 4],
                                #[2, 5], [2, 6], [3, 4], [3, 5], [3, 6], [4, 5],
                                #[4, 6], [5, 6]
                                ]):
    # We only take the two corresponding features
    X = transformed_data
    y = labelled_data

    # Train
    clf = DecisionTreeClassifier().fit(X, y)


    # Plot the decision boundary
    plt.subplot(2, 3, pairidx + 1)

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step),
                         np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel("PCA First Component")
    plt.ylabel("PCA Second Component")

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], c=color, cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

plt.suptitle("Decision surface of our PCA transformed data")
plt.legend(loc='lower right', borderpad=0, handletextpad=0)
plt.axis("tight")

plt.figure()
clf = DecisionTreeClassifier().fit(transformed_data[:,0], transformed_data[:,1])
plot_tree(clf, filled=True)
plt.show()

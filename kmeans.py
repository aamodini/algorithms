import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import datasets
from sklearn.decomposition import PCA

# import some data to play with
iris = datasets.load_iris()
iris_df = pd.DataFrame(data= np.c_[iris['data'], iris['target']],
                     columns= iris['feature_names'] + ['target'])

# always remember to wrap around with print otherwise there's no output
print(iris_df.head())

# attempt to create a class for kmean
# pseudo code for kmeans:
# 1. Identify the number of clusters k
# 2. Randomly split the dataset into K clusters and assign each subset a cluster label.
# This will initialise k number of centroids at random.
# 3. Calculate the cluster means
# 4. Assign each observation to the closest cluster mean. So if the distance between centroidA and point1 is greater
# then the disctance between centroidB and point1 then point1 should be reclassified to classB.
# 5. Continue iterating through steps 3 and 4 till the centroid no longer changes.

import numpy as np
import pandas as pd
from numpy.random import rand
import gzip
import json
import ast
from scipy.cluster.vq import kmeans2
#https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.kmeans2.html

import matplotlib.pyplot as plt
import random

#first we implement the 2 means algorithm from scipy so that we can divide A, the set of accounts into two parts as needed, so that we can
# recurse is reqd in algo 2 when we build the tree. We test the whole process with 10 to 20 points each of dimension 300, which are the 
# 10 to 20 accounts with 300 features. So our set A on which we run the tree building algorithm is essentially a collection of around 10^6 accounts
# each with 300 features, i.e. 10^6 points in 300 dimension.

#Later we have to use a more balanced mean, the algo 1 to be used in algo 2, that is 2 means but the sizes ofthe clusters are more
# more uniform. Now we first construct the tree with the inbuilt 2-means(k=2)

# For our purpose, we first test the 2 means algorithm on the matrix 10 x 300 which we generate from normal guassian dist. Then we
# partition the set A into 2 smaller sets depending on tau.

# the input is Mx300 matrix and we return two matrices M1x300 and M2x300 such that M1+M2 = M. In all the matrices, the rows are the points
# and the columns are the features. Since we are taking test input as guassian generated, no need to normalize the columns, else when we 
# actually deal with Amazon data, prior to applying kmeans, look at whiten.
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.vq.whiten.html#scipy.cluster.vq.whiten

# Generate the test set of accounts
mu = 0
sigma = 1.0
A = np.random.normal(mu, sigma, (10,300))

#Now do the 2-means and get the output.
centroid, label = kmeans2(A, 2, minit='points')
#centroid will be two arrays each of 300 dimension for the two clusters
# label[i] is the code or index of the centroid the ith observation is closest to.

# Now we write the function that partitons A into A0 and A1
def partition(A):
    centroid, label = kmeans2(A, 2, minit='points')
    A0,A1 = [],[]
    for i in range(10):
        if label[i] == 0:
            A0.append(A[i,:])
        else:
            A1.append(A[i,:])
    A0 = np.array(A0)
    A1 = np.array(A1)
    return (A0,A1)

# x,y = partition(A) this gives the two matrices as we want and now we can recurse on them if the number of rows in x or y is still
# greater than tau

#Now to create tree recursively, https://stackoverflow.com/questions/14084367/tree-in-python-recursive-children-creating
'''
r = Rectangle (100, 100)
r.splitUntilLevel (2)
print (r)

this will print the following

100 x 100
  100 x 32
    29 x 32
    71 x 32
  100 x 68
    32 x 68
    68 x 68
'''

class Node:
    def __init__ (self, data, parent = None):
        #self.width = width
        #self.height = height
        self.parent = parent
        self.children = []
        self.data = data

    @property
    def level (self):
        return 0 if not self.parent else 1 + self.parent.level

    def split (self):
        if self.children == []: return
        numrow,numcol = self.data.shape
        if numrow <= tau:
            self.children = []
        A1,A2 = partition(self.data)
        self.children = [Node(A1, self),Node(A2, self)]
        
    def splitUntilLevel (self, maxLevel):
        if maxLevel <= self.level: return
        self.split ()
        for child in self.children: child.splitUntilLevel (maxLevel)

    def __str__ (self):
        s = "{} x {}".format(self.data.size[0], self.data.size[1])
        for child in self.children: s += str (child)
        return s


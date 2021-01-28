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

# we test with bin matrices which has say 20 columns, this value should be 300
# we take 10 rows which is the number of accounts, this value should be around 10^6
mu = 0
sigma = 1.0
A = np.random.normal(mu, sigma, (10,10))
#Now do the 2-means and get the output.
centroid, label = kmeans2(A, 2, minit='points')
#centroid will be two arrays each of 10 dimension for the two clusters
# label[i] is the code or index of the centroid the ith observation is closest to.

# Now we write the function that partitons A into A0 and A1
# A0 and A1 are lists which will contain only the indices of the rows which are supposed to be in A0 and A1
# partition takes an arr which means it contains the matrix made up of A[arr[i]]
# we need to form matrix from the rows of A as mentioned in arr
arr = [0,1,2,3,4,5,6,7,8,9]
def partition(arr):
    matrix = []
    for i in range(len(arr)):
        matrix.append(A[i,:])
    matrix = np.array(matrix)
    centroid, label = kmeans2(matrix, 2, minit='points')
    A0,A1 = [],[]
    for i in range(len(arr)):# this is the num of the rows, i.e. the num of accts
        if label[i] == 0:
            A0.append(i)# means A[i] should be in this as a row
        else:
            A1.append(i)# means A[i] should be in this as a row
    #A0 = np.array(A0)
    #A1 = np.array(A1)
    return (A0,A1)# returns two lists so that x = A0[i], A[x] will give the reqd row

# x,y = partition(A) this gives the two matrices as we want and now we can recurse on them if the number of rows in x or y is still greater than tau
'''
x,y = partition(A)
x
This will print [0, 2, 3, 5, 7, 8], though this answer chnages everytime we run the code since Kmeans2 differs in the output everytime

Now to create tree recursively, https://stackoverflow.com/questions/14084367/tree-in-python-recursive-children-creating

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
    
We imitate the same idea here for building the tree
'''

tau = 3
# data contains lists which contain indices i. A[i] will be the reqd rows, to start with data = [0,...,9]
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
        numrow = len(self.data)
        if numrow <= tau:
            self.children = []
        A1,A2 = partition(self.data)
        self.children = [Node(A1, self),Node(A2, self)]
        
    def splitUntilLevel (self, maxLevel):
        if maxLevel <= self.level: return
        self.split ()
        for child in self.children: child.splitUntilLevel (maxLevel)

    def __str__ (self):#-----------------------------------------------this needs to be modified to print tree efficiently and check if algorithm 2 and 3 are working
        s = "{} x {}".format(self.data.size[0], self.data.size[1])
        for child in self.children: s += str (child)
        return s

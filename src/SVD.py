import random
import gzip
import json
import ast
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import lil_matrix
from numpy.random import rand
from scipy.sparse import csr_matrix, isspmatrix_csr

# making sure that the elems in each row and cols are unique will guarnatee that elems in matrix are bin
# %reset

# I tested the code with randomly generated bin matrices. Here is how I generate them

def createRandomSortedList(num, start, end): 
    arr = [] 
    tmp = random.randint(start, end) 
      
    for x in range(num): 
          
        while tmp in arr: 
            tmp = random.randint(start, end) 
              
        arr.append(tmp) 
    return arr 
myrow = createRandomSortedList(10, 1, 10000)
mycol = createRandomSortedList(10, 1, 10000)

row  = np.array(myrow)
col  = np.array(mycol)
newdata = np.array(np.ones(10))
data = np.array(newdata)
A = coo_matrix((data, (row, col)), shape=(10000, 10000)).tocsr()# this matrix A is the input to the svd

# Now the svd begins, look at the slide pg 2, in the folder for the algorithm

k = 300
l = k+20
m = 10000
n = 10000
# now in the first step we choose k = 100, l = 102, i = 2, A is considered mxn , so m = 10000, n= 5000
# In the first step, Using a random number generator, form a real n × l matrix G whose entries are independent and identically distributed
# Gaussian random variables of zero mean and unit variance
mu = 0
sigma = 1.0
G = np.random.normal(mu, sigma, (n,l))
#Compute B = AG (B ∈ R^ m×l )
#B = np.matmul(A,G)#---------------------------------------------------------first multiplication by A
B = []
for i in range(l):
    v = G[:,i]
    x = A.dot(v)
    B.append(x)
B = np.array(B).T
A = None # no more needed
G = None# not reqd anymore
X, lamda, Yt = np.linalg.svd(B, full_matrices=True)

lamda = None
Yt =  None
Q = X[:, : k] #(Q ∈ R m×k )
#Qt = Q.transpose()

#T = np.matmul(Qt,A)#-------------------------------2nd mult by A, only these two are there in this algo

At = coo_matrix((data, (col, row)), shape=(10000, 10000)).tocsr()
T = []
for i in range(k):
    v = Q[:,i]
    x = At.dot(v)
    T.append(x)
T = np.array(T)
At = None# not needed anymore
row = None
col = None
W, singlr, Vt = np.linalg.svd(T, full_matrices=True)
T = None# not needed anymore
Vt = None# not needed for the smaller dim matrix creation
U = np.matmul(Q,W)
Q = None
W = None
final = np.multiply(U,singlr)

# printing final with print the reqd matrix

#df = pd.DataFrame (final)
#filepath = 'output.xlsx'
#df.to_excel(filepath, index=False)

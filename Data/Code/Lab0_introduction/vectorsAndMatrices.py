# Vectors And Matrices
# Lorenzo Sforni
# Bologna, 21/09/2023

import numpy as np 
 
# ################################################################################################################
v = np.array([0,1,2,3])

print('The vector v is ')
print(v)
print(v[1:])

# Indexing/Slicing (e.g., on vectors)
print(v[0])
print(v[-1])
print(v[-2])
print(v[0:2])
print(v[1:])
print(v[1:-1])

v3 = np.random.randint(low=0,high=10,size=(10))
print('v3',v3)
print('nonzero indexes',np.nonzero(v3))
print('nonzero entries',v3[np.nonzero(v3)])

print('where entries',np.where(v3 < 4))  # -1 is broadcast
print('where entries',np.where(v3 < 4, v3, -1))  # -1 is broadcast 
# Where True, yield v3, otherwise yield -1.

# Hard copy
v = np.array([0, 1, 2, 3, 4, 5])
v_copy = np.copy(v) # v.copy()
v_copy[0] = 10
print(v_copy); print(v)

################################################ RANDOM quantities
# Uniform distribution
r = np.random.uniform(low=0.0, high=1.0, size=(4)) # size=(4,1)
print('A ud random vector is ', r)
print('with shape {}'.format(r.shape))
# Gaussian distribution
r = np.random.randn(4) # randn(4,1)
print('A gd random vector is ', r)
print('with shape {}'.format(r.shape))
# Uniform integer distribution
# randint(low, high=None, size=None)
n = 10
np.random.seed(0)
r = np.random.randint(low=0, high=10, size=(n))
print('A gd random vector is ', r)

############################################################# Matrices
# Define a matrix
A = np.array([
  [0,1,-2.3,0.1], # row 1
  [1.3, 4, -0.1, 0], 
  [4.1, -1.0, 0, 1.7]
  ])
print(A)
nrows,ncols = A.shape
# [nrows,ncols] = A.shape # Alternatives
# (nrows,ncols) = A.shape # Alternatives
print("Number of rows is {} \nNumber of columns is {}".format(nrows,ncols))
# Third column of A
A[:,2]
# Last column of A
A[:,-1]
# Second row of A, returned as vector
A[1,:]
M = A@A.T # square matrix
eigenvalues,eigenvectors = np.linalg.eig(M)
# EE = np.linalg.eig(M) # EE is a tuple
# eigenvalues,_ = np.linalg.eig(M)
print('Eigenvalues of M are:')
print(eigenvalues)
print('Eigenvectors of M are:')
print(eigenvectors)
print('Are they orthonormal?\n', eigenvectors.T@eigenvectors)
print('Diagonalization:\n', eigenvectors.T@M@eigenvectors - np.diag(eigenvalues))
# More on Linear Algebra
B = np.random.randn(4,4)
detB = np.linalg.det(B)
print('B is:', B)
print('Determinant of B is:')
print(detB)
print('{:.4f}'.format(detB)) # Prettier...
invB = np.linalg.inv(B); print('Inverse of B is: ', invB)
rankB = np.linalg.matrix_rank(B); print('Rank of B is: {}'.format(rankB))
traceB = np.trace(B); print('Trace of B is: {}'.format(traceB))
### Matrix multiplication
A = np.random.randn(4,4)
x = np.random.randn(4)
y = np.random.randn(4,1)
z = np.random.randn(1,4)
# Flatten...
yflat= y.flatten()
print('Shape of y: {} shape of yflat: {}'.format(y.shape,yflat.shape))
# y_ = x[:,None]
# Printing? Misleading...
print('\nx:',x,'\ny:',y,'\nz:',z,'\n')
# Chaos...
print((x-x[:,None]).shape)
print(x-x[:,None])    #Broadcasting issues...
sum_elements_A = np.sum(A) # np.mean(A)
print('The sum of the elements of A is', sum_elements_A)
sum_cols_A = np.sum(A,axis=0) # moves on the first axis
print('The sum of the columns of A is', sum_cols_A)
sum_rows_A = np.sum(A,axis=1) # moves on the second axis
print('The sum of the rows of A is', sum_rows_A)
### diag
# A = np.random.randint(5,size=(3,3))
diagA = np.diag(A)
print(A)
print(diagA)
print(np.diag(diagA))

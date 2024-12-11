import numpy

f = numpy.zeros((4, 10000, 50000))
print(f[:,:,0].shape)

xx = numpy.zeros((7, 1000))
print(xx[:,0+1].shape)

x = numpy.array([1, 2, 3, 4])
xx = numpy.array([1, 2, 3, 34])
print(x.shape)
print(x[:,None].shape)

Q=numpy.eye(4)
print(">")
print(((x.T-xx.T)@Q@(x-xx)+x.T@Q@x*1/2).squeeze().shape)
print((Q@(x-xx)).shape)
print(Q@xx)
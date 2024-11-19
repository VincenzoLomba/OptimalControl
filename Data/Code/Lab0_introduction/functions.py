# Functions
# Lorenzo Sforni
# Bologna, 21/09/2023

import numpy as np

def lin_comb(coef, vectors):
    """
        This is a function performing the linear combination of the $vectors via $coef
    """
    # print(type(vectors))
    n = len(vectors[0]) # Size of each vector element of vectors

    # the resulting
    out_vect = np.zeros(n)
    for i in range(len(vectors)): #Number of vectors to be combined
        out_vect = out_vect  + coef[i]*vectors[i]
    return out_vect 
#  def lin_comb(coef, vectors):
#     return sum(coef[i]*vectors[i] for i in range(len(vectors)))

v1 = np.array([1,2])
v2 = np.array([[3],[4]])
a1 = -0.5
a2 = 1.5
# print(a1*v1 + a2*v2)
w = lin_comb([a1, a2], [v1,v2]) # two lists ... 
print('{}'.format(w))
exit()

###############################################################################
###############################################################################

### Default Arguments
def student(firstname, lastname='Blu', title='Dr.'):
     print('The tutor is', title, firstname, lastname)
     # return None
     # return (firstname, lastname)
     return firstname, lastname
 
# 1 positional argument
out = student('Lorenzo')
print(out)
out1,out2 = student('Lorenzo')
print(out1, out2)
# 3 positional arguments  (Complete)
student('Lorenzo', 'Bianchi', 'Dr.')    
# 2 positional arguments
student('Mario', 'Rossi')
# 2 non-positional arguments...
student('Giovanni', title='Mr.')

# Imposed cost function
# Group 15
# Eugenio Piccini, 
# Mirko Legnini,
# Francesco Davide Bossio
import numpy as np
import Dynamics as dyn
xDim = dyn.xDim
uDim = dyn.uDim

QQt = 0.01*np.diag([0.1, 0.1, 100, 10, 100, 100])
RRt = 0.01*np.diag([5000, 0.00001])

QQT = QQt

def stagecost(xx,uu, xx_ref, uu_ref):

  # Stage-cost 

  # Quadratic cost function 
  # l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

  # Args
  #   - xx \in \R^2 state at time t
  #   - xx_ref \in \R^2 state reference at time t

  #   - uu \in \R^1 input at time t
  #   - uu_ref \in \R^2 input reference at time t


  # Return 
  #   - cost at xx,uu
  #   - gradient of l wrt x, at xx,uu
  #   - gradient of l wrt u, at xx,uu

  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  ll = 0.5*(xx - xx_ref).T @ QQt @ (xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

  dldx = QQt@(xx - xx_ref)
  dldu = RRt@(uu - uu_ref) #derivative of the quadratic form

  return ll.squeeze(), dldx, dldu

def termcost(xx,xx_ref):

    # Terminal-cost

    # Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    # Args
    #   - xx \in \R^2 state at time t
    #   - xx_ref \in \R^2 state reference at time t

    # Return 
    #   - cost at xx,uu
    #   - gradient of l wrt x, at xx,uu
    #   - gradient of l wrt u, at xx,uu

  xx = xx[:,None]
  xx_ref = xx_ref[:,None]

  llT = 0.5*(xx - xx_ref).T@QQT@(xx - xx_ref)

  lTx = QQT @ (xx - xx_ref) #derivative of terminal cost wrt x

  return llT.squeeze(), lTx
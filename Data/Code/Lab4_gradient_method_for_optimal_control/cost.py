#
# Gradient method for Optimal Control
# Cost functions
# Lorenzo Sforni, Marco Falotico
# Bologna, 22/11/2022
#

import numpy as np
import control
import dynamics as dyn

ns = dyn.ns
ni = dyn.ni

# QQt = np.array([[10000, 0], [0, 100]])
QQt = 0.1*np.diag([100.0, 1.0])
RRt = 0.01*np.eye(ni)
# RRt = 1*np.eye(ni)

# QQT = QQt


def stagecost(xx,uu, xx_ref, uu_ref):
  """
    Stage-cost 

    Quadratic cost function 
    l(x,u) = 1/2 (x - x_ref)^T Q (x - x_ref) + 1/2 (u - u_ref)^T R (u - u_ref)

    Args
      - xx \in \R^2 state at time t
      - xx_ref \in \R^2 state reference at time t

      - uu \in \R^1 input at time t
      - uu_ref \in \R^2 input reference at time t


    Return 
      - cost at xx,uu
      - gradient of l wrt x, at xx,uu
      - gradient of l wrt u, at xx,uu
  
  """

  xx = xx[:,None]
  uu = uu[:,None]

  xx_ref = xx_ref[:,None]
  uu_ref = uu_ref[:,None]

  ll = 0.5*(xx - xx_ref).T@QQt@(xx - xx_ref) + 0.5*(uu - uu_ref).T@RRt@(uu - uu_ref)

  lx = QQt@(xx - xx_ref)
  lu = RRt@(uu - uu_ref)

  return ll.squeeze(), lx, lu

def termcost(xT,xT_ref, QQT = QQt):
  """
    Terminal-cost

    Quadratic cost function l_T(x) = 1/2 (x - x_ref)^T Q_T (x - x_ref)

    Args
      - xT \in \R^2 state at time t
      - xT_ref \in \R^2 state reference at time t

    Return 
      - cost at xT,uu
      - gradient of l wrt x, at xT,uu
      - gradient of l wrt u, at xT,uu
  
  """

  # xT = xT[:,None]
  # xT_ref = xT_ref[:,None]

  llT = 0.5*(xT - xT_ref).T@QQT@(xT - xT_ref)

  lTx = QQT@(xT - xT_ref)


  return llT.squeeze(), lTx
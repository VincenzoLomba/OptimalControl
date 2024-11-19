#
# Finite-time LQR for regulation
# Discrete-time dynamics
# Lorenzo Sforni, Marco Falotico
# Bologna,  8/11/2024
#

import numpy as np

# discretization step

dt = 1e-3

def dynamics(xx,uu):
  """
    Dynamics of a discrete-time mass spring damper system

    Args
      - xx \in \R^2 state at time t
      - uu \in \R^1 input at time t

    Return 
      - next state xx_{t+1}
      - gradient of f wrt x, at xx,uu
      - gradient of f wrt u, at xx,uu
    
    Note
      - AA = dxf'
      - BB = duf' 
  """

  ns = 2
  ni = 1
         
  bb = 1
  mm = 1
  kspring = 1e1

  xx = xx.squeeze()
  uu = uu.squeeze()
    
    
  # Mass-spring-damper cart
  # x1_d = x2
  # x2_d = -kspring / mm x1 - bb/mm x2 + 1/mm u

  xx_plus = np.zeros((ns, ));
  
  xx_plus[0] = xx[0] + dt * xx[1];
  xx_plus[1] = xx[1] + dt * (-kspring / mm * xx[0] - bb / mm * xx[1] + uu / mm);

  #Gradient

  dxf = np.array([[1, -dt * kspring / mm],
      [dt, 1 + dt * (- bb / mm)]])

  duf = np.array([[0, dt * 1 / mm]])

  return xx_plus, dxf, duf




#
#Gradient Method for Optimal Control
#Reference Trajectory Generation
#Marco Falotico
#Bologna, 8/11/2024
#


import numpy as np
import dynamics as dyn

from scipy.linalg import solve_discrete_are
from scipy.integrate import solve_ivp



def gen(step_reference, tf, dt, ns, ni):

      TT = int(tf/dt)
      
      ref_deg_0 = 0
      ref_deg_T = 30

      
      xx_ref = np.zeros((ns, TT))
      uu_ref = np.zeros((ni, TT))

      
      if not step_reference:
            ...
            #TODO, try generating a smooth trajectory

      else:

            KKeq = dyn.KKeq

            xx_ref[0,int(TT/2):] = np.ones((1,int(TT/2)))*np.ones((1,int(TT/2)))*np.deg2rad(ref_deg_T)
            uu_ref[0,:] = KKeq*np.sin(xx_ref[0,:])
   
      return xx_ref, uu_ref
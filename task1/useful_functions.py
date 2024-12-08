import numpy.linalg as la
import numpy as np
from Auxiliaries.costs import *
from Auxiliaries.final_dyn import *

def is_pos_def(M, tol=1e-8):
    #Check symmetry of matrix M with tolerance tol=1e-8
    if not np.allclose(M, M.T, atol=tol):
        M = 0.5*(M+M.T)   #Forcing the simmetry
    
    #Check if M is pos def using Cholensky decomposition: M pos.def. symmetric --> M = L.T*L
    try: 
        la.cholesky(M)
        return True
    except:
        la.LinAlgError
        return False
    
# TRAJECTORIES GENERATORS:

#1) STEP/RAMP GENERATOR
def traj_gen(step_reference, tf, dt, ns, ni):

      TT = int(1/dt)
      #TT = int(100)
      
      ref_deg_0 = 0
      ref_deg_T = 30

      
      xx_ref = np.zeros((ns, TT))
      uu_ref = np.zeros((ni, TT))

      
      if not step_reference:
            # In this case we generate a ramp along all the time horizon
            xx_ref[0,:] = np.linspace(np.deg2rad(ref_deg_0), np.deg2rad(ref_deg_T), TT)
            uu_ref[0,:] = np.sin(xx_ref[0,:])

      else:
            xx_ref[0,int(TT/2):] = np.ones((1,int(TT/2)))*np.ones((1,int(TT/2)))*np.deg2rad(ref_deg_T)
            uu_ref[0,:] = np.sin(xx_ref[0,:])
   
      return xx_ref, uu_ref
                  print("WARNING: no stepsize was found with armijo rule!")

      return stepsize

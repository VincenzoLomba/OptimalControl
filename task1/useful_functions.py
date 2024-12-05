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
    
def stepsize_armijo(stepsize_0, armijo_maxiters, cc, beta, deltau, xx_ref, uu_ref,  x0, uu, JJ, descent_arm):

      """
      Computes the stepsize using Armijo's rule.
      input parameters:
            - stepsize_0 : initial stepsize guess,
            - armijo_maxiters : maximum number of iterations for armijo rule,
            - deltau : descending direction for the control action,
            - xx_ref : reference trajectory state,
            - uu_ref : reference trajectory input,
            - x0 : initial state,
            - uu : input at current iteration,
            - JJ : cost at current iteration,
            - descent_arm: armijo descent direction at current iteration,
            
      output parameters:
            - stepsize
      """

      TT = uu.shape[1]


      stepsizes = []  # list of stepsizes
      costs_armijo = []

      stepsize = stepsize_0

      ns = xx_ref.shape[0]
      ni = uu_ref.shape[0]


      for ii in range(armijo_maxiters):

            # temp solution update

            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))

            xx_temp[:,0] = x0


            for tt in range(TT-1):
                  uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
                  xx_temp[:,tt+1] = discretizedDynamicFRA(xx_temp[:,tt], uu_temp[:,tt])[0].squeeze()

            # temp cost calculation
            JJ_temp = 0

            for tt in range(TT-1):
                  temp_cost = stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                  JJ_temp += temp_cost

            temp_cost = termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
            JJ_temp += temp_cost

            stepsizes.append(stepsize)      # save the stepsize
            costs_armijo.append(np.min([JJ_temp, 100*JJ]))    # save the cost associated to the stepsize

            if JJ_temp > JJ  + cc*stepsize*descent_arm:
                  # update the stepsize
                  stepsize = beta*stepsize

            else:
                  print('Armijo stepsize = {:.3e}'.format(stepsize))
                  break
            
            if ii == armijo_maxiters -1:
                  print("WARNING: no stepsize was found with armijo rule!")

      return stepsize

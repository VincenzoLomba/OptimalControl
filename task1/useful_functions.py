import numpy.linalg as la
import numpy as np
from cost import *
from NL_dynamics_FRA import *
from scipy.linalg import solve_discrete_are
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


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

def traj_gen(reference_type, tf, dt, ns, ni, eq_start, eq_end):
    # Numero di step temporali
    TT = int(tf/dt)
    
    # Array conversion
    eq_start = np.array(eq_start).reshape(ns)  # eq_start: array di dim (ns,)
    eq_end = np.array(eq_end).reshape(ns)      # eq_end: array di dim (ns,)

    # Allocazione memoria per le traiettorie
    xx_ref = np.zeros((ns, TT))
    uu_ref = np.zeros((ni, TT))

    time = np.linspace(0, tf, TT)
    
    if reference_type == 'ramp':
        # Genera una rampa per ogni stato
        # Da eq_start[i] a eq_end[i]
        for i in range(ns):
            xx_ref[i, :] = np.linspace(eq_start[i], eq_end[i], TT)

    elif reference_type == 'step':
        # A metà tempo passa da eq_start a eq_end per ogni stato
        half_T = TT // 2
        xx_ref[:, :half_T] = eq_start[:, np.newaxis]  # shape (ns, half_T)
        xx_ref[:, half_T:] = eq_end[:, np.newaxis]    # shape (ns, TT-half_T)

    elif reference_type == 'sigmoid':
        # Genera una traiettoria sigmoidale da eq_start a eq_end
        midpoint = tf / 2
        steepness = 10
        for i in range(ns):
            xx_ref[i, :] = eq_start[i] + (eq_end[i]-eq_start[i]) / (1 + np.exp(-steepness*(time - midpoint)))
    else:
        raise ValueError("Invalid reference_type. Choose 'step', 'ramp', or 'sigmoid'.")

    # Generazione di uu_ref
    # Esempio: se ni == ns, u è un vettore con la stessa dimensione di stato.
    # Qui, per semplicità, definiamo uu_ref come sin degli stati.
    # Se ni != ns, si adatta di conseguenza. Per esempio se ni=1, prendiamo il seno del primo stato.
    if ni == ns:
        uu_ref = np.sin(xx_ref)
    else:
        # Se ni=1, ad esempio:
        uu_ref[0, :] = np.sin(xx_ref[0, :])

    # Debugging opzionale
    # print(f"Generated {reference_type} trajectory from {eq_start} to {eq_end}:")
    # print("xx_ref:", xx_ref)
    # print("uu_ref:", uu_ref)

    return xx_ref, uu_ref




def tot_cost(xx, uu, xx_ref, uu_ref, TT):
    JJ = 0
    for tt in range(TT-1): 
      temp_cost = stagecost(xx[:,tt], uu[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
      JJ += temp_cost
    
    temp_cost = termcost(xx[:,-1], xx_ref[:,-1])[0]
    JJ += temp_cost
    return JJ





def solver_costate_eq(xx, uu, xx_ref, uu_ref, TT):
    # In here we compute lambdas using AA, BB linearized dynamics and we also compute dJ and qq, rr cost affine terms:
    lmbd = zeros_like(xx)
    AAt = zeros((ns,ns,TT))
    BBt = zeros((ns,ni,TT))
    dJ = zeros_like(uu)
    qqt = zeros_like(xx_ref)
    rrt = zeros_like(uu_ref)

    qqT = termcost(xx[:,TT-1], xx_ref[:,TT-1])[1].squeeze()
    lmbd[:,TT-1] = qqT

    for tt in reversed(range(TT-1)): #integration backwards in time
        aat, bbt, _, _, _ = stagecost(xx[:,tt], uu[:,tt], xx_ref[:,tt], uu_ref[:,tt])[1:]

        dfdx, dfdu, _, _, _, _ = discretizedDynamicFRA(xx[:,tt], uu[:,tt])[1:]
        AA = dfdx.T
        BB = dfdu.T

        lmbd_temp = AA.T@lmbd[:,tt+1][:,None] + aat # costate equation
        dJ_temp = BB@lmbd[:,tt+1][:,None] + bbt # gradient of J wrt u

        lmbd[:,tt] = lmbd_temp.squeeze()
        dJ[:,tt] = dJ_temp.squeeze()

        qqt[:,tt] = aat.squeeze()
        rrt[:,tt] = bbt.squeeze()
        AAt[:,:,tt] = AA
        BBt[:,:,tt] = BB.T
    return AAt, BBt, qqt, rrt, qqT, dJ, lmbd


####### ARMIJO #######

def stepsize_armijo(stepsize_0, armijo_maxiters, cc, beta, deltau, xx_ref, uu_ref,  x0, uu, JJ, TT, descent_arm, plot=False):
      stepsizes = []  # list of stepsizes
      costs_armijo = []
      stepsize = stepsize_0


      for ii in range(armijo_maxiters):

            # temp solution update

            xx_temp = np.zeros((ns,TT))
            uu_temp = np.zeros((ni,TT))

            xx_temp[:,0] = x0


            for tt in range(TT-1):
                  uu_temp[:,tt] = uu[:,tt] + stepsize*deltau[:,tt]
                  xx_temp[:,tt+1] = dyn.discretizedDynamicFRA(xx_temp[:,tt], uu_temp[:,tt])[0]

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
            
            
      ############################
      # Descent Plot
      ############################

      if plot:

            steps = np.linspace(0,stepsize_0,int(2e1))
            costs = np.zeros(len(steps))

            for ii in range(len(steps)):

                  step = steps[ii]

                  # temp solution update

                  xx_temp = np.zeros((ns,TT))
                  uu_temp = np.zeros((ni,TT))

                  xx_temp[:,0] = x0

                  for tt in range(TT-1):
                        uu_temp[:,tt] = uu[:,tt] + step*deltau[:,tt]
                        xx_temp[:,tt+1] = dyn.dynamics(xx_temp[:,tt], uu_temp[:,tt])[0]

                        # temp cost calculation
                  JJ_temp = 0

                  for tt in range(TT-1):
                        temp_cost = stagecost(xx_temp[:,tt], uu_temp[:,tt], xx_ref[:,tt], uu_ref[:,tt])[0]
                        JJ_temp += temp_cost

                  temp_cost = termcost(xx_temp[:,-1], xx_ref[:,-1])[0]
                  JJ_temp += temp_cost

                  costs[ii] = np.min([JJ_temp, 100*JJ])


            plt.figure(1)
            plt.clf()

      
            plt.plot(steps, costs, color='g', label='$J(\\mathbf{u}^k - stepsize*d^k)$')
            plt.plot(steps, JJ + descent_arm*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            # plt.plot(steps, JJ - descent*steps, color='r', label='$J(\\mathbf{u}^k) - stepsize*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')
            plt.plot(steps, JJ + cc*descent_arm*steps, color='g', linestyle='dashed', label='$J(\\mathbf{u}^k) - stepsize*c*\\nabla J(\\mathbf{u}^k)^{\\top} d^k$')

            plt.scatter(stepsizes, costs_armijo, marker='*') # plot the tested stepsize

            plt.grid()
            plt.xlabel('stepsize')
            plt.legend()
            plt.draw()

            plt.show()

      return stepsize

from Auxiliaries.final_dyn import *
from Auxiliaries.parameters_3 import *
from numpy import *
from Auxiliaries.ltv_LQR_affine import ltv_LQR
from Auxiliaries.costs import *
from Auxiliaries.useful_functions import *

def NM_robustOC(xx0, uu0, xx_ref, uu_ref, TT, max_iter=50, tol=1e-4):
    """
    Newton Method for optimal control with closed loop, regularized approach.

    Args:
        xx0 (array): State initial system (dimension ns).
        uu0 (array): Initial input (dimension ni x TT-1).
        TT (int): Time horizon.
        max_iter (int): Maximum number of iterations.
        tol (float): Convergence tolerance.

    Returns:
        xx_traj: Optimal state trajectory.
        uu_traj: Optimal control trajectory.
        cost (list): Cost list per iteration.
    """

    # Initializing variables
    xx_traj = zeros((ns, TT))
    xx_traj[:, 0] = xx0
    uu_traj = uu0.copy()
    cost_history = []

    for k in range(max_iter):
        # Forward Simulation with the actual trajectory (x_traj, u_traj)
        for t in range(TT - 1):
            xx_traj[:, t + 1] = discretizedDynamicFRA(xx_traj[:, t], uu_traj[:, t])[0].squeeze()
            

        # Computation of the actual cost
        total_cost = 0
        for t in range(TT - 1):
            ll, _, _, _, _, _ = stagecost(xx_traj[:, t], uu_traj[:, t], xx_ref[:, t], uu_ref[:, t])
            total_cost += ll
        llT, _, _ = termcost(xx_traj[:, -1], xx_ref[:, -1])
        total_cost += llT
        cost_history.append(total_cost)
        print(total_cost)

        # Verifying convergence
        if k > 0 and abs(cost_history[-1] - cost_history[-2]) < tol:
            print(f"Convergence obtained in {k} iterations.")
            break
        
        #COMPUTATION OF DESCENT DIRECTION:
        #STEP 1.1: Computation of AAt, BBt, qqt, rrt (and declaration of QQt, RRt, SSt & their regularized version)
        AAt = zeros((ns, ns, TT - 1))
        BBt = zeros((ns, ni, TT - 1))
        QQt = zeros((ns, ns, TT))
        RRt = zeros((ni, ni, TT - 1))
        SSt = zeros((ni, ns, TT - 1))
        QQt_reg = zeros((ns, ns))
        RRt_reg = zeros((ni, ni))
        SSt_reg = zeros((ni, ns))
        qqt = zeros((ns, TT))
        rrt = zeros((ni, TT - 1))

        for t in range(TT - 1):
            _, AA, BB, d2fdxdx, d2fdxdu, _, d2fdudu = discretizedDynamicFRA(xx_traj[:, t], uu_traj[:, t])
            _, lx, lu, lxx, lxu, luu = stagecost(xx_traj[:, t], uu_traj[:, t], xx_ref[:, t], uu_ref[:, t])
            AAt[:, :, t] = AA
            BBt[:, :, t] = BB
            qqt[:, t] = lx.squeeze()
            rrt[:, t] = lu.squeeze()

        # Defining the terminal values of QQ and qq:
        _, lTx, lTxx = termcost(xx_traj[:, -1], xx_ref[:, -1])
        QQt[:, :, -1] = lTxx
        qqt[:, -1] = lTx
        QQT = QQt[:, :, -1] 
        qqT = qqt[:, -1]
        
        #STEP 1.2: Compute the costates (lambda_t)
        lambda_t = zeros((ns, TT))
        lambda_t[:, -1] = qqT  # Terminal condition for lambda_T

        for t in reversed(range(TT - 1)):
            lambda_t[:, t] = AAt[:, :, t].T @ lambda_t[:, t + 1]

        #We define their regularized version
        QQt_reg = lxx
        RRt_reg = luu
        SSt_reg = lxu

        for t in range(TT - 1):
            #STEP 1.3 Defining matrices QQt, RRt, SSt:
            #Here we implement a regularization: we check if QQt, RRt, SSt are pos.def.; YES--> we use them ELSE --> we use their reg version
            contractionQQt = np.tensordot(d2fdxdx, lambda_t[:, t + 1], axes=([0], [0]))
            contractionRRt = np.tensordot(d2fdudu, lambda_t[:, t + 1], axes=([0], [0]))
            contractionSSt = np.tensordot(d2fdxdu, lambda_t[:, t + 1], axes=([0], [0])) 
            
            QQt[:, :, t] = lxx + contractionQQt
            RRt[:, :, t] = luu + contractionRRt
            SSt[:, :, t] = lxu + contractionSSt.T

            QQt[:, :, t] = QQt[:, :, t] if is_pos_def(QQt[:, :, t]) else QQt_reg
            RRt[:, :, t] = RRt[:, :, t] if is_pos_def(RRt[:, :, t]) else RRt_reg
            SSt[:, :, t] = SSt[:, :, t] if is_pos_def(SSt[:, :, t]) else SSt_reg
        
        #DEBUG
        print(f"AAt shape: {AAt.shape}")      # Dovrebbe essere (ns, ns, TT - 1)
        print(f"BBt shape: {BBt.shape}")      # Dovrebbe essere (ns, ni, TT - 1)
        print(f"QQt shape: {QQt.shape}")      # Dovrebbe essere (ns, ns, TT)
        print(f"RRt shape: {RRt.shape}")      # Dovrebbe essere (ni, ni, TT - 1)
        print(f"SSt shape: {SSt.shape}")      # Dovrebbe essere (ni, ns, TT - 1)
        print(f"QQT shape: {QQT.shape}")      # Dovrebbe essere (ns, ns)
        print(f"ni: {ni}, ns: {ns}, TT: {TT}")


        #STEP 1.4: conclusion --> We compute the Gain K and the cost sigma to compute delta_uu and finally delta_xx
        KK, sigma, PP = ltv_LQR(AAt, BBt, QQt, RRt, SSt, QQT, TT, xx0, qqt, rrt, qqT)

        #STEP 2: Update of control sequence 
        xx_new = zeros_like(xx_traj)
        uu_new = zeros_like(uu_traj)
        delta_uu = zeros_like(uu_traj)

        #Definition of Stepsize Armijo rule's parameters
        descent_arm = 0.0
        stepsize_0 = 1
        armijo_iters = 20
        cc = 0.5
        beta = 0.7
        gamma = 1 #it will be updated with Armijo

        for t in range(TT - 1):
            #Control sequence update: 
            delta_uu[:, t] = KK[:, :, t] @ (xx_new[:, t] - xx_traj[:, t]) + sigma[:, t]
            descent_arm = delta_uu[:, t].T @ rrt[:, t]
        
        gamma = stepsize_armijo(stepsize_0, armijo_iters, cc, beta, delta_uu, xx_ref, uu_ref, xx0, uu_traj, total_cost, descent_arm)
        
        for t in range(TT - 1):
            uu_new[:, t] = uu_traj[:, t] + KK[:, :, t] @ (xx_new[:, t] - xx_traj[:, t]) + gamma * sigma[:, t]
    
            #Compute the new state using the dynamics: 
            xx_new[:, t + 1] = discretizedDynamicFRA(xx_new[:, t], uu_new[:, t])[0].squeeze()
            


        #STEP 3: TRAJECTORY UPDATE
        xx_traj = xx_new
        uu_traj = uu_new

    return xx_traj, uu_traj, cost_history

    


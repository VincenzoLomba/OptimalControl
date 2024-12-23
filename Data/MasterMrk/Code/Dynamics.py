# Dynamics of the system to control
# Group 15
# Eugenio Piccini, 
# Mirko Legnini,
# Francesco Davide Bossio

# x -> xx[0]
# y -> xx[1]
# psi -> xx[2]
# V -> xx[3]
# beta -> xx[4]
# psi-dot -> xx[5]

# delta -> uu[0]
# Fx -> uu[1]
import numpy as np

mm = 1480
Iz = 1950
aa = 1.421
bb = 1.029
mu = 1
gg = 9.81

dt = 0.01

xDim = 6
uDim = 2

Fzf = (mm * gg * bb)/(aa + bb)
Fzr = (mm * gg * aa)/(aa + bb)

def integrate_position(xx): ##integrate x, y and psi given V, beta and psi
    xx_positions_next=np.zeros((int(xDim/2)))
    xx_positions_next[0] = xx[0] + dt * xx[3] * np.cos(xx[4] + xx[2])
    xx_positions_next[1] = xx[1] + dt * xx[3] * np.sin(xx[4] + xx[2])
    xx_positions_next[2] = xx[2] + dt * xx[5]
    return xx_positions_next

def dynamics_equilibrium (xx, uu): ##dynamics parametrized for the equilibrium    
    #x[0] -> delta 
    #x[1] -> Fx
    #x[2] -> psi-dot
    
    # fixed V=u[0] and beta = u[1]
        
    if xx[0] != 0:
        betaF = xx[0] - ((uu[0] * np.sin(uu[1]) + aa * xx[2]) / (uu[0] * np.cos(uu[1])))
        betaR = - ((uu[0] * np.sin(uu[1]) - bb * xx[2]) / (uu[0] * np.cos(uu[1])))
    else:
         betaF = 0
         betaR = 0
    
    Fyf = mu * Fzf * betaF
    Fyr = mu * Fzr * betaR
    
    xdot = np.zeros((3,1))
    
    #Dynamics
    
    xdot[0] = (Fyr * np.sin(uu[1]) + xx[1] * np.cos(uu[1] - xx[0]) + Fyf * np.sin(uu[1] - xx[0])) / mm
    

    if uu[0] != 0:
         xdot[1] = ((Fyr*np.cos(uu[1])+Fyf*np.cos(uu[1]-xx[0])-xx[1]*np.sin(uu[1]-xx[0]))/(mm*uu[0])) - xx[2]
    else:
         xdot[1] = 0
        
    xdot[2] = (((xx[1] * np.sin(xx[0]) + Fyf * np.cos(xx[0])) * aa) - Fyr * bb) / Iz

    xdot = np.squeeze(xdot)
    
    return xdot

def dynamics (xx, uu):##full system dynamics, returns next position and gradients wrt x and u 
    xx = xx[:,None]
    uu = uu[:,None]
    if xx[3,0] != 0:
        betaF = uu[0,0] - ((xx[3,0] * np.sin(xx[4,0]) + aa * xx[5,0]) / (xx[3,0] * np.cos(xx[4,0])))
        betaR = - ((xx[3,0] * np.sin(xx[4,0]) - bb * xx[5,0]) / (xx[3,0] * np.cos(xx[4,0])))
    else:
         betaF = 0
         betaR = 0
    
    Fyf = mu * Fzf * betaF
    Fyr = mu * Fzr * betaR
    
    xNext = np.zeros((xDim,1))                                                 

    xNext[0,0] = xx[0,0] + dt * xx[3,0] * np.cos(xx[4,0] + xx[2,0])
    xNext[1,0] = xx[1,0] + dt * xx[3,0] * np.sin(xx[4,0] + xx[2,0])
    xNext[2,0] = xx[2,0] + dt * xx[5,0]

    xNext[3,0] = xx[3,0] + dt * ((Fyr * np.sin(xx[4,0]) + uu[1,0] * np.cos(xx[4,0] - uu[0,0]) + Fyf * np.sin(xx[4,0] - uu[0,0])) / mm)
    if xx[3,0] != 0:
        xNext[4,0] = xx[4,0] + dt * (((Fyr * np.cos(xx[4,0]) + Fyf * np.cos(xx[4,0] - uu[0,0]) - uu[1,0] * np.sin(xx[4,0] - uu[0,0])) / (mm * xx[3,0])) - xx[5,0])
    else:
        xNext[4,0] = 0

    xNext[5,0] = xx[5,0] + dt * (((uu[1,0] * np.sin(uu[0,0]) + Fyf * np.cos(uu[0,0])) * aa) - Fyr * bb) / Iz

    # Initialize gradient wrt x 
    Nabla1 = np.zeros((6,6))
    
    #derive x
    Nabla1[0,0]=1;
    Nabla1[1,0]=0;
    Nabla1[2,0]=dt*(-xx[3,0]*np.sin(xx[4,0] + xx[2,0]));
    Nabla1[3,0]=dt*(np.cos(xx[4,0] + xx[2,0]));
    Nabla1[4,0]=dt*(-xx[3,0]*np.sin(xx[4,0] + xx[2,0]));
    Nabla1[5,0]=0;

    #derive y
    Nabla1[0,1]=0;
    Nabla1[1,1]=1;
    Nabla1[2,1]=dt*(xx[3,0]*np.cos(xx[4,0] + xx[2,0]));
    Nabla1[3,1]=dt*(np.sin(xx[4,0] + xx[2,0]));
    Nabla1[4,1]=dt*(xx[3,0]*np.cos(xx[4,0] + xx[2,0]));
    Nabla1[5,1]=0;

    #derive psi
    Nabla1[2,2]=1
    Nabla1[5,2] = dt

    #derive V 
    Nabla1[0, 3]=0; 
    Nabla1[1, 3]=0;
    Nabla1[2, 3]=0;
    Nabla1[3, 3]=1+dt*(-(Fzf*mu*np.sin(xx[4,0] - uu[0,0])*(np.sin(xx[4,0])/(xx[3,0]*np.cos(xx[4,0])) - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]**2*np.cos(xx[4,0]))) + (Fzr*mu*np.sin(xx[4,0])**2)/(xx[3,0]*np.cos(xx[4,0])) + (Fzr*mu*np.sin(xx[4,0])*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0])))/(xx[3,0]**2*np.cos(xx[4,0])))/mm);
    Nabla1[4, 3]=dt*(-(uu[1,0]*np.sin(xx[4,0] - uu[0,0]) + Fzr*mu*np.sin(xx[4,0]) - Fzf*mu*np.cos(xx[4,0] - uu[0,0])*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0]))) + Fzf*mu*np.sin(xx[4,0] - uu[0,0])*((np.sin(xx[4,0])*(aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0])))/(xx[3,0]*np.cos(xx[4,0])**2) + 1) - (Fzr*mu*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0])))/xx[3,0] - (Fzr*mu*np.sin(xx[4,0])**2*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0])))/(xx[3,0]*np.cos(xx[4,0])**2))/mm);
    Nabla1[5, 3]=dt*((Fzr*bb*mu*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0])) - (Fzf*aa*mu*np.sin(xx[4,0] - uu[0,0]))/(xx[3,0]*np.cos(xx[4,0])))/mm;

    #derive beta
    Nabla1[0,4]=0;
    Nabla1[1,4]=0;
    Nabla1[2,4]=0;
    Nabla1[3,4]=dt*(-((Fzr*mu*np.sin(xx[4,0]))/xx[3,0] + Fzf*mu*np.cos(xx[4,0] - uu[0,0])*(np.sin(xx[4,0])/(xx[3,0]*np.cos(xx[4,0])) - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]**2*np.cos(xx[4,0]))) + (Fzr*mu*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0])))/xx[3,0]**2)/(xx[3,0]*mm) - (Fzf*mu*np.cos(xx[4,0] - uu[0,0])*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0]))) - uu[1,0]*np.sin(xx[4,0] - uu[0,0]) + (Fzr*mu*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0])))/xx[3,0])/(xx[3,0]**2*mm));
    Nabla1[4,4]=1+dt*(-(uu[1,0]*np.cos(xx[4,0] - uu[0,0]) + Fzr*mu*np.cos(xx[4,0]) + Fzf*mu*np.sin(xx[4,0] - uu[0,0])*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0]))) + Fzf*mu*np.cos(xx[4,0] - uu[0,0])*((np.sin(xx[4,0])*(aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0])))/(xx[3,0]*np.cos(xx[4,0])**2) + 1))/(xx[3,0]*mm));
    Nabla1[5,4]=dt*(((Fzr*bb*mu)/xx[3,0] - (Fzf*aa*mu*np.cos(xx[4,0] - uu[0,0]))/(xx[3,0]*np.cos(xx[4,0])))/(xx[3,0]*mm) - 1);
    
    #derive psi dot 
    Nabla1[0, 5]=0;
    Nabla1[1, 5]=0;
    Nabla1[2, 5]=0;
    Nabla1[3, 5]=dt*(((Fzr*bb*mu*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0])) - Fzf*aa*mu*np.cos(uu[0,0])*(np.sin(xx[4,0])/(xx[3,0]*np.cos(xx[4,0])) - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]**2*np.cos(xx[4,0]))) + (Fzr*bb*mu*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0])))/(xx[3,0]**2*np.cos(xx[4,0])))/Iz)
    Nabla1[4, 5]=dt*(-(Fzf*aa*mu*np.cos(uu[0,0])*((np.sin(xx[4,0])*(aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0])))/(xx[3,0]*np.cos(xx[4,0])**2) + 1) - Fzr*bb*mu + (Fzr*bb*mu*np.sin(xx[4,0])*(bb*xx[5,0] - xx[3,0]*np.sin(xx[4,0])))/(xx[3,0]*np.cos(xx[4,0])**2))/Iz)
    Nabla1[5, 5]=1+dt*(-((Fzf*mu*np.cos(uu[0,0])*aa**2)/(xx[3,0]*np.cos(xx[4,0])) + (Fzr*mu*bb**2)/(xx[3,0]*np.cos(xx[4,0])))/Iz)

    #Initialize gradient wrt u
    Nabla2=np.zeros((2,6))

    ##Derive V 
    Nabla2[0,3]=dt*((uu[1,0]*np.sin(xx[4,0] - uu[0,0]) + Fzf*mu*np.sin(xx[4,0] - uu[0,0]) - Fzf*mu*np.cos(xx[4,0] - uu[0,0])*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0]))))/mm);
    Nabla2[1,3]=dt*(np.cos(xx[4,0] - uu[0,0])/mm);

    #Derive beta
    Nabla2[0,4]=dt*((uu[1,0]*np.cos(xx[4,0] - uu[0,0]) + Fzf*mu*np.cos(xx[4,0] - uu[0,0]) + Fzf*mu*np.sin(xx[4,0] - uu[0,0])*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0]))))/(xx[3,0]*mm));
    Nabla2[1,4]=dt*(-np.sin(xx[4,0] - uu[0,0])/(xx[3,0]*mm))
    
    #Derive psi dot
    Nabla2[0,5]=dt*((aa*(uu[1,0]*np.cos(uu[0,0]) + Fzf*mu*np.cos(uu[0,0]) - Fzf*mu*np.sin(uu[0,0])*(uu[0,0] - (aa*xx[5,0] + xx[3,0]*np.sin(xx[4,0]))/(xx[3,0]*np.cos(xx[4,0])))))/Iz)
    Nabla2[1,5]=dt*((aa*np.sin(uu[0,0]))/Iz)

    return np.squeeze(xNext), Nabla1, Nabla2
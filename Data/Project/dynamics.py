# Bologna,  28/11/2024
# Flexible Robotic Arm Discretized Dynamics

from numpy import *
from parameters import *

dt = dtCollection.task0_discretizationStep # definition of the discretization step

def discretizedDynamicFRA(xx,uu):
    """
    Discretized Dynamics of the Flexible Robotic Arm
    Arguments:
    - xx: 4x1 column vector state at time t x=[x0, x1, x2, x3]'=[ϑ1, ϑ2, dϑ1dt, dϑ2dt]'
    - uu: 1x1 input value at time t
    Returns:
    - xxp: 4x1 column vector state at time t+1
    - dfdx: 4x4 jacobian of the dynamics wrt x at xx,uu
    - dfdu: 4x1 jacobian of the dynamics wrt u at xx,uu
    """
    xx = xx.squeeze()
    uu = uu.squeeze()

    # Inertia matrix (2x2)
    M = array([
        [I1+I2+ m1*(r1)**2+m2*((l1)**2+(l2)**2)+2*m2*l1*r2*cos(xx[1]),  I2+m2*(r2)**2+m2*l1*r2*cos(xx[1]) ],
        [I2+m2*(r2)**2+m2*l1*r2*cos(xx[1]),                             I2+m2*(r2)**2                     ]
    ])
    # Coriolis and centrifugal forces matrix (2x1)
    C = array([
        -m2*l1*r2*xx[3]*sin(xx[1])*(xx[3]+2*xx[2]),
        m2*l1*r2*sin(xx[1])*(xx[2])**2
    ])
    # Gravity forces matrix (2x1)
    G = array([
        g*(m1*r1+m2*l1)*sin(xx[0])+g*m2*r2*sin(xx[0]+xx[1]), 
        g*m2*r2*sin(xx[0]+xx[1])
    ])
    # Friction forces matrix (2x2)
    F = array([
        [f1, 0], 
        [0, f2]
    ])
    # Input matrix (2x1)
    U = array([uu, 0])

    # Derivative wrt the second state x1 of the inertia matrix (2x2)
    dMdx1 = array([
        [-2*m2*l1*r2*sin(xx[1]),    -m2*l1*r2*sin(xx[1]) ], 
        [-m2*l1*r2*sin(xx[1]),      0                    ]
    ])
    # Tensor derivative wrt the state vector of the intertia matrix (2x2x4)
    dMdx = [zeros((2,2)), dMdx1, zeros((2,2)), zeros((2,2))]
    # Inverse of the inertia matrix (2x2)
    invM = linalg.pinv(M)
    # Derivative wrt the second state x1 of the inverse of the inertia matrix (2x2)
    dinvMdx1 = -invM*dMdx1*invM

    # Tensor derivative wrt the state vector of the Coriolis and centrifugal forces matrix (2x1x4)
    dCdx0 = array([0, 0])
    dCdx1 = array([
        -m2*l1*r2*xx[3]*cos(xx[1])*(xx[3]+2*xx[2]),
        m2*l1*r2*cos(xx[1])*(xx[2])**2
    ])
    dCdx2 = array([
        -2*m2*l1*r2*xx[3]*sin(xx[1]), 
        2*m2*l1*r2*sin(xx[1])*(xx[2])
    ])
    dCdx3 = array([
        -m2*l1*r2*xx[3]*sin(xx[1])-m2*l1*r2*sin(xx[1])*(xx[3]+2*xx[2]),
        0
    ])
    dCdx = array([dCdx0, dCdx1, dCdx2, dCdx3])

    # Tensor derivative wrt the state vector of the gravity forces matrix (2x1x4)
    dGdx0 = array([
        g*(m1*r1+m2*l1)*cos(xx[0])+g*m2+r2*cos(xx[0]+xx[1]),
        g*m2*r2*cos(xx[0]+xx[1])
    ])
    dGdx1 = array([
        g*m2+r2*cos(xx[0]+xx[1]), 
        g*m2*r2*cos(xx[0]+xx[1])
    ])
    dGdx2 = array([0, 0])
    dGdx3 = array([0, 0])
    dGdx = array([dGdx0, dGdx1, dGdx2, dGdx3])

    # Tensor derivative wrt the state vector of the friction forces matrix (2x2x4)
    # dFdx = [zeros((2,2)), zeros((2,2)), zeros((2,2)), zeros((2,2))]

    # Definition of the 4x1 column vector state at time t+1 according to the discretized dynamics
    # Notice that this is the definition of the discretized dynamic function f=[f0, f1, f2, f3]'
    xxp = zeros((ns, 1))
    xxp[0] = xx[0] + dt*xx[2]
    xxp[1] = xx[1] + dt*xx[3]
    xxp[2:4] = (xx[2:4] + dt*invM@(U-G-F@array([xx[2], xx[3]])-C)).reshape(2,1)

    # Jacobian of the dynamics wrt x at xx,uu (dfdx, 4x4) (notice f=[f0, f1, f2, f3]')
    # Notice that the minor 2x2 extracted from dfdx at last two rows and first two columns if the Jacobian of [f2, f3]' wrt [x0, x1]'
    # Notice that the minor 2x2 extracted from dfdx at last two rows and last two columns if the Jacobian of [f2, f3]' wrt [x2, x3]'
    # Notice that [f2, f3]'=invM(x1)*(U-G(x0,x1)-F*[x2, x3]'-C(x0,x1,x2,x3))
    dfdx = zeros((ns, ns))
    dfdx[0, :] = [1, 0, dt, 0]
    dfdx[1, :] = [0, 1, 0, dt]
    dfdx[2:4, 0] = dt*(invM@(-dGdx[0]-dCdx[0]))
    dfdx[2:4, 1] = dt*(invM@(-dGdx[1]-dCdx[1]) + dinvMdx1@(U-F@array([xx[2], xx[3]]-C-G)))
    dfdx[2:4, 2] = array([1, 0]) + dt*(invM@(-dGdx[2]-F@array([1, 0])-dCdx[2]))
    dfdx[2:4, 3] = array([0, 1]) + dt*(invM@(-dGdx[3]-F@array([0, 1])-dCdx[3]))

    # Jacobian of the dynamics wrt u at xx,uu (dudx, 4x1) (notice f=[f0, f1, f2, f3]')
    dfdu = zeros((ns, ni))
    dfdu[2:4, 0] = dt*invM@array([1, 0])

    return xxp, dfdx, dfdx
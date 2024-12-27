# Flexible Robotic Arm Discretized Dynamics

from parameters import *
from numpy import *
from sympy import symbols, Matrix, lambdify, hessian, diff, sin, cos
from datetime import datetime

dt = discretizationStep # definition of the discretization step (loading from parameters)

def discretizedDynamicFRA():
    """
    Discretized Dynamics of the Flexible Robotic Arm
    Arguments:
    - xx: 4x1 column vector state at time t x=[x0, x1, x2, x3]'=[ϑ1, ϑ2, dϑ1dt, dϑ2dt]' (from a python variable p.o.f., this is of shape (4,))
    - uu: 1x1 input value at time t (from a python variable p.o.f., this is oh shape (1,))
    Returns:
    - xxp: 4x1 column vector state at time t+1 (from a python variable p.o.f., this is of shape (4,))
    - dfdx: 4x4 jacobian of the dynamics wrt x at xx,uu
    - dfdu: 4x1 jacobian of the dynamics wrt u at xx,uu
    - d2fdxdx: 4x4x4 hessian of the dynamics wrt x two times at xx,uu
    - d2fdxdu: 4x4x1 hessian of the dynamics one time wrt x and one time wrt u at xx,uu
    - d2fdudx: 4x1x4 hessian of the dynamics one time wrt u and one time wrt x at xx,uu
    - d2fdudu: 4x1x1 hessian of the dynamics wrt u two times at xx,uu
    """

    # Simbols definition
    theta1, theta2, dtheta1, dtheta2, u = symbols('theta1 theta2 dtheta1 dtheta2 u', real=True)
    thetas = [theta1, theta2, dtheta1, dtheta2]

    # Inertia matrix M(theta2) (2x2)
    M = Matrix([
        [I1+I2+ m1*(r1)**2+m2*((l1)**2+(r2)**2)+2*m2*l1*r2*cos(theta2),  I2+m2*(r2)**2+m2*l1*r2*cos(theta2) ],
        [I2+m2*(r2)**2+m2*l1*r2*cos(theta2),                             I2+m2*(r2)**2                      ]
    ])
    # Coriolis and centrifugal forces matrix C(theta1, theta1, dtheta1, dtheta2) (2x1)
    C = Matrix([
            -m2*l1*r2*dtheta2*sin(theta2)*(dtheta2+2*dtheta1),
            m2*l1*r2*sin(theta2)*(dtheta1)**2
    ])
    # Gravity forces matrix G(theta1, theta2) (2x1)
    G = Matrix([  
        g*(m1*r1+m2*l1)*sin(theta1)+g*m2*r2*sin(theta1+theta2), 
        g*m2*r2*sin(theta1+theta2)
    ])
    # Friction forces matrix (2x2)
    F = Matrix([
        [f1, 0], 
        [0, f2]
    ])
    # Control input vector (2x1)
    U = Matrix([u, 0])

    # Definition of the 4x1 column vector state at time t+1 according to the discretized dynamics
    # Notice that this is the definition of the discretized dynamic function f=[f0, f1, f2, f3]'
    xxp = Matrix([
        theta1 + dt*dtheta1,
        theta2 + dt*dtheta2,
        Matrix([dtheta1, dtheta2]) + dt*M.inv()*(U-G-F*Matrix([dtheta1, dtheta2])-C)
    ])

    # Jacobian of the dynamics wrt x at xx,uu (dfdx, 4x4) (notice f=[f0, f1, f2, f3]')
    dfdx = xxp.jacobian(thetas)

    # Jacobian of the dynamics wrt u at xx,uu (dfdu, 4x1) (notice f=[f0, f1, f2, f3]')
    dfdu = xxp.jacobian([u])

    # Tensor Hessian of the dynamics wrt x two times at xx,uu (d2fdxdx, 4x4x4)
    # Tensor hessian of the dynamics wrt u two times at xx,uu (d2fdu2, 4x1x1)
    # Tensor hessian of the dynamics wrt x one time and u one time at xx,uu (d2fdxdu, 4x4x1)
    # Tensor hessian of the dynamics wrt u one time and x one time at xx,uu (d2fdudx, 4x1x4)
    d2fdxdx = []
    d2fdudu = []
    d2fdxdu = []
    d2fdudx = []
    for i in range(ns):
        d2fdxdx.append(hessian(xxp[i], thetas))
        d2fdudu.append(hessian(xxp[i], [u]))
        d2fidxdu = []
        for j in range(ns):
            dfidxj = diff(xxp[i], thetas[j])
            d2fidxjdu = diff(dfidxj, u)
            d2fidxdu.append(d2fidxjdu)
        d2fdxdu.append(Matrix(d2fidxdu))
        dfidu = diff(xxp[i], u)
        d2fidudx = []
        for j in range(ns):
            d2fidudxj = diff(dfidu, thetas[j])
            d2fidudx.append(d2fidudxj)
        d2fdudx.append(Matrix(d2fidudx))

    xxpFunction = lambdify((theta1, theta2, dtheta1, dtheta2, u), xxp)
    dfdxFunction = lambdify((theta1, theta2, dtheta1, dtheta2, u), dfdx)
    dfduFunction = lambdify((theta1, theta2, dtheta1, dtheta2, u), dfdu)
    d2fdxdxFunction = lambdify((theta1, theta2, dtheta1, dtheta2, u), d2fdxdx)
    d2fdxduFunction = lambdify((theta1, theta2, dtheta1, dtheta2, u), d2fdxdu)
    d2fdudxFunction = lambdify((theta1, theta2, dtheta1, dtheta2, u), d2fdudx)
    d2fduduFunction = lambdify((theta1, theta2, dtheta1, dtheta2, u), d2fdudu)

    return lambda xx, uu: ([ 
        xxpFunction(*xx.squeeze(), uu.squeeze()).squeeze(),
        dfdxFunction(*xx.squeeze(), uu.squeeze()),
        dfduFunction(*xx.squeeze(), uu.squeeze()),
        d2fdxdxFunction(*xx.squeeze(), uu.squeeze()),
        d2fdxduFunction(*xx.squeeze(), uu.squeeze()),
        d2fdudxFunction(*xx.squeeze(), uu.squeeze()),
        d2fduduFunction(*xx.squeeze(), uu.squeeze())
    ])

def runDynamicFunction(discretizedDynamicFuntion, uu, xx0, TT):
    """
    Generic implementation of a forward-in-time evolution (in an open loop fashon) of the dynamic of a certain system
    Arguments:
    - discretizedDynamicFuntion: functions of (xx_t, uu_t) that implements the discretized dynamics of the system that is being considered,
                                 requiring as arguments respectively the state and input values at time t,
                                 returning the state value at time t+1 AND all the jacobians and hessians of the dynamics wrt state and input in the following order:
                                 xxp, dfdx, dfdu, d2fdxdx, d2fdxdu, d2fdudx, d2fdudu
    - uu_des: column vector input curve (from a python variable p.o.f., this is of shape (ni,TT))
    - xx0: column vector initial state (from a python variable p.o.f., this is of shape (ns,))
    - TT: number of time steps (each one of duration dt, enough for evolve from t=0 to t=T, where [0, T] is the considered horizon)
    Returns:
    - xx: column vector state trajectory obtained by running the dynamic of the system (from a python variable p.o.f., this is of shape (ns,TT))
    """

    xx = zeros((len(xx0), TT))
    xx[:,0] = xx0 # (notice: this is an assignment between shaped (ns,) python objects)
    for tt in range(TT-1):
        xx[:,tt+1] = discretizedDynamicFuntion(xx[:,tt], uu[:,tt])[0] # (notice: this is an assignment between shaped (ns,) python objects)
    return xx

def dynamicC(xx):
    """ Given the state vector xx, this function returns the Coriolis and centrifugal forces matrix (2x1) """
    return array([
        -m2*l1*r2*xx[3]*sin(xx[1])*(xx[3]+2*xx[2]),
        m2*l1*r2*sin(xx[1])*(xx[2])**2
    ])

def dynamicG(xx):
    """ Given the state vector xx, this function returns the gravity forces matrix (2x1) """
    return array([
        g*(m1*r1+m2*l1)*sin(xx[0])+g*m2*r2*sin(xx[0]+xx[1]), 
        g*m2*r2*sin(xx[0]+xx[1])
    ])
# Flexible Robotic Arm Task1: from a desired smooth trajectory (that evolves from one equilibrium to another)
#                             obtain an optimal one thanks to the regularized Newton [like] Method (in its closed-loop version)


import parameters as params
from trajectories import pascalSnailFRAPositionTrajectory
from numpy import diag, zeros, eye, array, squeeze, ones, sin, cos
from methods import runNewtonMethodTrkTrj
from dynamics import discretizedDynamicFRA as discretizedDynamicFuntion
from matplotlib import pyplot

def task2():

    dt = params.discretizationStep;
    T = params.TCollection.task2_trajectoryDuration;
    ns = params.ns
    ni = params.ni

    TT = int(T/dt) # number of time steps (each one of duration dt, enough for evolve from t=0 to t=T)
    deltsSnail = T/4*3

    # Getting the desired trajectory (in terms of evolution of the two angles of the FRA)
    x0, x1, t = pascalSnailFRAPositionTrajectory(T, deltsSnail, dt)
    xx_des = zeros((ns, TT))
    xx_des[0,:] = x0
    xx_des[1,:] = x1
    uu_des = 31.08112533*ones((ni, TT))

    # Notice that in order to plot the trajectory as evolution in the plane of the End Effector of the FRA,
    # you should use the following equations:
    # x = params.r1*sin(xx_des[0,:]) + params.r2*sin(xx_des[0,:]+xx_des[1,:])
    # y = - params.r1*cos(xx_des[0,:]) - params.r2*cos(xx_des[0,:]+xx_des[1,:])
    # pyplot.figure()
    # pyplot.plot(x, y)
    # pyplot.grid()
    # pyplot.show()

    # Defining cost matrices (for a trajectory tracking optimization problem) and applying the Newton Method
    QQ = diag([100, 100, 1, 1])
    #RR = zeros((ni, ni, TT))
    #RRzeroSlice = int(TT/16)
    #RR[:,:,0:RRzeroSlice] = 100*eye(ni)
    #RR[:,:,RRzeroSlice:TT-RRzeroSlice] = 0.01*eye(ni)
    #RR[:,:,TT-RRzeroSlice:TT] = 100*eye(ni)
    RR = 0.01*eye(ni)
    QQT = 10**6*eye(ns)
    newtonMethodMaxIterations = 60
    xx_opt, uu_opt = runNewtonMethodTrkTrj(
        xx_des, uu_des, xx_des[:, 0], TT, newtonMethodMaxIterations,
        discretizedDynamicFuntion, 1e-4,
        QQ, RR, QQT, None
    )

    # Plotting results
    pyplot.close('all')
    pyplot.pause(1)
    pyplot.figure()
    for i in range(0, ns): pyplot.plot(squeeze(t), squeeze(array(xx_des[i,:])), label='ϑ'+str(i)+' desired')
    pyplot.legend(); pyplot.show(block=False); pyplot.pause(0.5)
    pyplot.plot(squeeze(t), squeeze(uu_des), label='u desired')
    pyplot.legend(); pyplot.show(block=False); pyplot.pause(0.5)
    _, ax = pyplot.subplots()
    for i in range(ns):
        line_des, = ax.plot(squeeze(t), squeeze(xx_des[i, :]), label='ϑ' + str(i) + ' desired')
        color = line_des.get_color()
        ax.plot(squeeze(t), squeeze(xx_opt[i, :]), '--', color=color, label='ϑ' + str(i) + ' optimal')
    ax.legend(loc='upper left')
    pyplot.show(block=False); pyplot.pause(0.5)
    _, ax = pyplot.subplots()
    line_u_des, = ax.plot(squeeze(t), squeeze(uu_des), label='u desired')
    color_u = line_u_des.get_color()
    ax.plot(squeeze(t), squeeze(uu_opt[0, :]), '--', color=color_u, label='u optimal')
    ax.legend()
    pyplot.show();

    return xx_opt, uu_opt

if __name__ == "__main__":
    task2()
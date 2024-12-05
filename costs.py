from numpy import * 
from Auxiliaries.parameters_3 import *

QQt = 0.1*array([[100.0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
RRt = eye(ni, ni)

def stagecost(xx, uu, xx_ref, uu_ref):
    xx = xx[:,None]
    uu = uu[:,None]

    xx_ref = xx_ref[:,None]
    uu_ref = uu_ref[:,None]

    delta_xx = xx - xx_ref
    delta_uu = uu - uu_ref

    '''
    #Debug
    print("Forma delta_xx:", delta_xx.shape)
    print("Forma QQt:", QQt.shape)
    print("Forma delta_uu:", delta_uu.shape)
    print("Forma RRt:", RRt.shape)
    '''

    ll = 0.5*delta_xx.T@QQt@delta_xx + 0.5*delta_uu.T@RRt@delta_uu

    lx = QQt@(xx - xx_ref)
    lu = RRt@(uu - uu_ref)

    lxx = QQt
    lxu = zeros((ni, ns))
    luu = RRt

    return ll.squeeze(), lx, lu, lxx, lxu, luu

def termcost(xxT, xxT_ref, QQT=QQt):

    llT = 0.5*(xxT - xxT_ref).T@QQT@(xxT - xxT_ref)

    lTx = QQT@(xxT - xxT_ref)

    lTxx = QQT

    return llT.squeeze(), lTx, lTxx
#
# Discrete-time LQR solvers
# Lorenzo Sforni, Marco Falotico  
# Bologna, 8/11/2024
#

import numpy as np


def lti_LQR(AA, BB, QQ, RR, QQf, TT):

  """
	LQR for LTI system with fixed cost	
	
  Args
    - AA (nn x nn) matrix
    - BB (nn x mm) matrix
    - QQ (nn x nn), RR (mm x mm) stage cost
    - QQf (nn x nn) terminal cost
    - TT time horizon
  Return
    - KK (mm x nn x TT) optimal gain sequence
    - PP (nn x nn x TT) riccati matrix
  """
	
  ns = AA.shape[1]
  ni = BB.shape[1]

  
  PP = np.zeros((ns,ns,TT))
  KK = np.zeros((ni,ns,TT))
  
  PP[:,:,-1] = QQf
  
  # Solve Riccati equation
  for tt in reversed(range(TT-1)):
    QQt = QQ
    RRt = RR
    AAt = AA
    BBt = BB
    PPtp = PP[:,:,tt+1]
    
    PP[:,:,tt] = QQt + AAt.T@PPtp@AAt - \
        + (AAt.T@PPtp@BBt)@np.linalg.inv((RRt + BBt.T@PPtp@BBt))@(BBt.T@PPtp@AAt)
  
  # Evaluate KK
  
  
  for tt in range(TT-1):
    QQt = QQ
    RRt = RR
    AAt = AA
    BBt = BB
    PPtp = PP[:,:,tt+1]
    
    KK[:,:,tt] = -np.linalg.inv(RRt + BBt.T@PPtp@BBt)@(BBt.T@PPtp@AAt)

  return KK, PP
    
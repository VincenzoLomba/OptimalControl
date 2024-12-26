
from methods import ComputeLocalLin, GenerateNoise, SolveLQPwithNoise

def task3():
    # Computation of local linearization around optimal trajectory and consequently LQR gain KK
    KK = ComputeLocalLin(xx_opt, uu_opt, QQ, RR, QQ, TT, discretizedDynamicFunction, solveLinearLQP)[0]

    # Design of the LQR with noise
    noise = GenerateNoise(xx_opt, noise_std_percentage=0.2)
    xx_track, uu_track = SolveLQPwithNoise(xx_opt, uu_opt, KK, noise, TT, discretizedDynamicFunction, False)
    xx_track, uu_track = SolveLQPwithNoise(xx_opt, uu_opt, KK, noise, TT, discretizedDynamicFunction, True)

if __name__ == "__main__":
    task3()
# Bologna,  30/11/2024
# Flexible Robotic Arm Task1: from a desired smooth trajectory to an optimal one
# thanks to the regularized Newton's like Method (in its closed-loop version)

from Auxiliaries.parameters_3 import *
from numpy import *
from Auxiliaries.newton_method_OC import *
from Auxiliaries.final_dyn import *
from Auxiliaries.costs import *
from Auxiliaries.equilibria import getEquilibriumPoints
from Auxiliaries.ltv_LQR_affine import ltv_LQR
from Auxiliaries.useful_functions import *
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

dt = dtCollection.task0_discretizationStep;
T = TCollection.task1_trajectoryDuration;

TT = int(1/dt) # number of time steps (each one of duration dt, enough for evolve from t=0 to t=T)

# Definition of the desired trajectory (step from eq. point [0, 0, 0, 0]' to eq. point [pi, 0, 0, 0]')
# Step 1: Calculate equilibrium points
print("Computing equilibrium points...")
x_eq1 = getEquilibriumPoints(array([0]), array([0, 0, 0, 0]))
x_eq2 = getEquilibriumPoints(array([0]), array([pi, 0, 0, 0]))

# Step 2: Build desired trajectory
xx_des = zeros((ns, TT))  # State reference trajectory
uu_des = zeros((ni, TT))  # Input reference trajectory

# Define durations for constant parts and transition
T_const = int(0.25 * TT)  # Constant segments near equilibriums
T_trans = TT - 2 * T_const  # Transition segment duration
t_const = linspace(0, T_const * dt, T_const)
t_trans = linspace(T_const * dt, (T_const + T_trans) * dt, T_trans)

# Constant parts
xx_des[:, :T_const] = tile(x_eq1, (T_const, 1)).T
xx_des[:, -T_const:] = tile(x_eq2, (T_const, 1)).T

# Smooth transition
for i in range(ns):
    # Correggiamo i nodi temporali in ordine crescente
    time_nodes = [0, T_const * dt, (T_const + T_trans) * dt, T * dt]

    # Assicuriamoci che i nodi siano strettamente crescenti
    time_nodes = sorted(list(set(time_nodes)))  # Elimina duplicati e ordina

    print(f"Nodi spline per stato {i}: {time_nodes}")  # Debug dei nodi temporali

    # Costruzione della spline con nodi corretti
    spline = CubicSpline(
        time_nodes,  # Nodi temporali
        [x_eq1[i], x_eq1[i], x_eq2[i], x_eq2[i]]  # Valori corrispondenti agli stati
    )

    # Generazione della traiettoria tra T_const e T_const + T_trans
    xx_des[i, T_const:T_const + T_trans] = spline(t_trans)


# Step 3: Apply Newton Method for Optimal Control

print("Running Newton's Method for Optimal Control...")
xx_opt, uu_opt, cost_history = NM_robustOC(x_eq1, uu_des[:, 0:TT - 1], xx_des, uu_des, TT)

# Step 4: Plot results
time_horizon = linspace(0, T, TT)

plt.figure(figsize=(10, 8))

# Plot state trajectories
for i in range(ns):
    plt.subplot(ns + ni, 1, i + 1)
    plt.plot(time_horizon, xx_des[i, :], 'g--', label="Desired trajectory")
    plt.plot(time_horizon, xx_opt[i, :], 'b', label="Optimal trajectory")
    plt.ylabel(f"$x_{i+1}$")
    plt.legend()
    plt.grid()

# Plot control trajectories
for i in range(ni):
    plt.subplot(ns + ni, 1, ns + i + 1)
    plt.plot(time_horizon, uu_des[i, :], 'r--', label="Desired control")
    plt.plot(time_horizon[:-1], uu_opt[i, :], 'k', label="Optimal control")
    plt.ylabel(f"$u_{i+1}$")
    plt.xlabel("Time [s]")
    plt.legend()
    plt.grid()

plt.tight_layout()
plt.show()

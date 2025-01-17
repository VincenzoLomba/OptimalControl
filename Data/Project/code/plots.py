
# Project Plotter

from methods import TrjTrkOCPData
from numpy import squeeze, full, nan
from matplotlib.cm import get_cmap
from matplotlib.pyplot import subplots, tight_layout, title, grid, show, figure, plot, xlabel, ylabel, legend

# def plotStateInputCurves(xxFirst, uuFirst, xxSecond, uuSecond, labelFirst, labelSecond):
def plotStateInputCurves(data: TrjTrkOCPData):
    """
    Generates ns subplots for desired and optimal state trajectories
    and a separate plot for the desired and optimal input trajectory
    """
    ns, _, t = data.ns, data.ni, data.t
    xx_des, uu_des = squeeze(data.xx_des), squeeze(data.uu_des)
    xx_opt, uu_opt = data.getOptimalTrajectory()

    # Color map definition (for the different states)
    colorMap = get_cmap('tab10', data.ns)
    # Creating various subplots (one for each state)
    _, axes = subplots(int(ns/2), 2, figsize=(10, 8))
    axes = axes.flatten()  # Axes flattening for a simpler indexing
    for i in range(ns):
        ax = axes[i]
        color = colorMap(i)
        ax.plot(t, xx_des[i, :], color = color, label = f'ϑ{i+1} desired')
        ax.plot(t, xx_opt[i, :], '--', color = color, label = f'ϑ{i+1} optimal')
        ax.set_title(f'Theta {i}')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Time')
        ax.set_ylabel('Value')
    tight_layout()
    show()

    figure(figsize=(8, 6))
    line, = plot(t, uu_des, label='Desired input')
    color = line.get_color()
    plot(t, uu_opt[0, :], '--', color=color, label='Optimal input')
    title('Input Trajectory')
    xlabel('Time')
    ylabel('Value')
    legend()
    grid(True)
    show()
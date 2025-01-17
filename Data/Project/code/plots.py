
# Project Plotter

from numpy import squeeze, linspace, full, nan
from matplotlib.cm import get_cmap
from matplotlib.pyplot import subplots, tight_layout, title, grid, show, figure, plot, xlabel, ylabel, legend

def plotStateInputCurves(xxFirst, uuFirst, xxSecond, uuSecond, labelFirst, labelSecond, dt):
    """
    Generates ns subplots for desired and optimal state trajectories and a separate plot for the desired and optimal (supposed single) input trajectory
    """
    TT = xxFirst.shape[1]
    ns, tx = xxFirst.shape[0], squeeze(linspace(0, dt*TT, TT))
    uuFirst = uuFirst[:, :-1]
    uuSecond = uuSecond[:, :-1]
    xxFirst, uuFirst = squeeze(xxFirst), squeeze(uuFirst)
    xxSecond, uuSecond = squeeze(xxSecond), squeeze(uuSecond)
    tu = tx[:-1]

    # Color map definition (for the different states)
    colorMap = get_cmap('tab10', ns)
    # Creating various subplots (one for each state)
    xFigure, axes = subplots(int(ns/2), 2, figsize=(10, 8))
    axes = axes.flatten()  # Axes flattening for a simpler indexing
    for i in range(ns):
        ax = axes[i]
        color = colorMap(i)
        ax.plot(tx, xxFirst[i, :], '--', color = color, label = f'ϑ{i+1} {labelFirst}')
        ax.plot(tx, xxSecond[i, :], color = color, label = f'ϑ{i+1} {labelSecond}')
        ax.set_title(f'Theta{i} Trajectory')
        ax.grid(True)
        ax.legend()
        ax.set_xlabel('Time (s)')
        ax.set_ylabel("Value (" + ("rad" if i < ns/2 else "rad/s") + ")")
    tight_layout()
    show()

    uFigure = figure(figsize=(8, 6))
    line, = plot(tu, uuFirst, '--', label=f'{labelFirst} input')
    color = line.get_color()
    plot(tu, uuSecond, color=color, label=f'{labelSecond} input')
    title('Input Trajectory')
    xlabel('Time (s)')
    ylabel('Value (Nm)')
    legend()
    grid(True)
    show()

    return xFigure, uFigure
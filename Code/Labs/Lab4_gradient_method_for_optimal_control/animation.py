
#Animation 

from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt

def animate_pendolum(xx_star, xx_ref, dt):
      """
      Animates the pendolum dynamics
      input parameters:
            - Optimal state trajectory xx_star
            - Reference trajectory xx_ref
            - Sampling time dt
      oputput arguments:
            None
      """

      TT = xx_star.shape[1]

      # Set up the figure and axis for the animation
      fig, ax = plt.subplots()
      ax.set_xlim(-2, 2)  # adjust limits as needed for pendulum's reach
      ax.set_ylim(-2, 2)

      # Plot elements
      pendulum_line, = ax.plot([], [], 'o-', lw=3, color="blue", label="Optimal Path")
      reference_line, = ax.plot([], [], 'o--', lw=2, color="green", label="Reference Path")
      time_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)

      ax.legend()
      ax.set_title("Pendulum Optimal Control Trajectory")
      ax.set_xlabel("X position")
      ax.set_ylabel("Y position")

      # Initial setup function for the animation
      def init():
            pendulum_line.set_data([], [])
            reference_line.set_data([], [])
            time_text.set_text('')
            return pendulum_line, reference_line, time_text

      # Update function for each frame of the animation
      def update(frame):
            # Pendulum position (optimal solution)
            x_opt = np.sin(xx_star[0, frame])  # assuming xx_star[0] is angle theta
            y_opt = -np.cos(xx_star[0, frame])
            
            # Reference position
            x_ref = np.sin(xx_ref[0, frame])
            y_ref = -np.cos(xx_ref[0, frame])

            # Update pendulum line
            pendulum_line.set_data([0, x_opt], [0, y_opt])
            reference_line.set_data([0, x_ref], [0, y_ref])

            # Update time text
            time_text.set_text(f'time = {frame*dt:.2f}s')

            return pendulum_line, reference_line, time_text

      # Create the animation
      ani = FuncAnimation(fig, update, frames=TT, init_func=init, blit=True, interval=0.1)

      # Display the animation
      plt.show()

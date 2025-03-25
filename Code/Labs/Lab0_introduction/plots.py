# Plot
# Lorenzo Sforni
# Bologna, 21/09/2023

import matplotlib.pyplot as plt
import numpy as np

x = [1, 8]
y = [3, 10]

plt.scatter(x, y)
plt.plot(x, y)
plt.plot(x, y, 'o', label='my line')
plt.title("A title...", loc = 'left')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid()
plt.show()

x = np.linspace(0, 4*np.pi,100)
y = np.sin(x)
y_ = np.cos(x)

fig = plt.figure()
ax = fig.add_subplot(111)
# ax = plt.subplot(111) # just one line...

ax.plot(x, y, '--', color='blue', linewidth=3, label='$\sin(x)$')
ax.plot(x, y_, ',-', color='red', linewidth=2.5, label='cos(x)')
ax.grid()
ax.set_xlim(1, 6.5)
ax.set_xlabel("Axis $x$")
ax.legend()
# plt.savefig('./Desktop/filename.png')
plt.show()

x = [1,2,3,4]
y = [10,20,25,30]

fig = plt.figure()
ax = fig.add_subplot(111)
# ax = plt.subplot(111) # just one line...

# PLOT 1
ax.plot(x, y, color='blue', linewidth=3)
# PLOT 2
ax.scatter(x,y,color='r')
# PLOT 3
ax.scatter(
            [2,4,6],
            [5,15,25],
            color='green',
            s=1e3,
            marker='^',
            )
ax.set_xlim(1, 6.5)
ax.grid()
ax.legend(loc='lower right')
ax.set_title("My title")
ax.set_xlabel("Axis x")
ax.set_ylabel("Axis y")

# plt.savefig('./Desktop/filename.png')
plt.show()

ReLU = lambda x: np.maximum(0, x)
x = np.arange(start=-10,stop=10,step=1)
# x = np.linspace(start=-10,stop=10, num=21)
y = ReLU(x)
plt.plot(x, y, 'r-', linewidth=3)
plt.grid()
plt.show()

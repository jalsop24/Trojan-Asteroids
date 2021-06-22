
import numpy as np
import matplotlib.pyplot as plt


print("Loading data...")

data = np.loadtxt("data.csv").transpose()

print("Done.")

t = data[0]
x = data[1]
y = data[2]
z = data[3]
energy = data[4]


axesLimits = 8

############
# Plotting #
############

fig = plt.figure()

# plt.plot(t, energy, label="Total Energy")

# plt.ylim(-25, 4)

# plt.xlabel("Time / Years")
# plt.ylabel("Total Energy / Arbitrary Units")

plt.plot(x, y, label="Relative Position")

plt.xlabel("x / AU")
plt.ylabel("y / AU")

# ax = fig.add_subplot(projection="3d", azim=-35, elev=30)
# ax.plot(xArray if inertial else sol.y[0], yArray if inertial else sol.y[1], sol.y[2], label="Relative Position")
# ax.set_zlim(-axesLimits, axesLimits)

plt.xlim(-axesLimits, axesLimits)
plt.ylim(-axesLimits, axesLimits)

plt.legend(loc="best")
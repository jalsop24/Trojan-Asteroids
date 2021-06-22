
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


axesLimits = 2

############
# Plotting #
############

fig = plt.figure()

plt.plot(x, y, label="Relative Position")

plt.xlabel("x / AU")
plt.ylabel("y / AU")

plt.xlim(-axesLimits, axesLimits)
plt.ylim(-axesLimits, axesLimits)

plt.legend(loc="best")
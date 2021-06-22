
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

yMax = -3
yMin = -5

############
# Plotting #
############

fig = plt.figure()

plt.plot(t, energy, label="Total Energy")

plt.ylim(yMin, yMax)

plt.xlabel("Time / Years")
plt.ylabel("Total Energy / Arbitrary Units")


plt.legend(loc="best")
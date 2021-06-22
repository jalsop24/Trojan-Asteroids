

import numpy as np
from solver import solver


####################
# Solver Constants #
####################

method = "Radau" # Radau, DOP853 # RK45 Works with rotating frame but not inertial

T_Min = 0

T_Max = 1000

Resolution = 500

inertial = False



######################
# Physical Constants #
######################

G = 4*np.pi**2 # Solar system units: unit time = 1 year, unit length = 1 AU



##########################
# Sun, Jupiter Constants #
##########################

alpha = 0.00000032

Msun = 1
Mjup = alpha * Msun

Rjup = 1.5 # Separation between Jupiter and the Sun

w_jup = np.sqrt( G*Msun/(Rjup**3) ) # Angular velocity of Jupiter around the CoM



###############################
# Asteroid Initial Conditions #
###############################

asteroidAngle = np.pi/2 - np.pi/180 * 60

RAst = 1.5

x0 = RAst * np.cos(asteroidAngle)
y0 = RAst * np.sin(asteroidAngle)
z0 = 0

theta0 = 0 * np.pi/180 

orbitalSpeed =  w_jup * Rjup * 1.0

vx0 = -orbitalSpeed*np.sin(asteroidAngle + theta0)
vy0 = orbitalSpeed*np.cos(asteroidAngle + theta0)
vz0 = 0

# Form the lists to feed to the solver giving both methods equivalent initial conditions

inertial_y0 = [x0, y0, z0, vx0, vy0, vz0]

rotating_y0 = [x0, y0, z0, vx0 + w_jup*y0, vy0 - w_jup*x0, vz0]

initialConditions = inertial_y0 if inertial else rotating_y0



#################################################
# Use the solver.py module to solve the system  #
#################################################


sol = solver(
            initialConditions, 
            method=method, 
            T_Min=T_Min, 
            T_Max=T_Max, 
            inertial=inertial, 
            Msun=Msun, 
            Mjup=Mjup, 
            Rjup=Rjup,
            Resolution=Resolution
            )



#########################################################
# Normalise the results from the two different methods  #
#########################################################

print("Transforming coordinates...")

# All operations and functions are in a vectorised form

t = sol.t
    
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]
    
vx = sol.y[3] 
vy = sol.y[4]
vz = sol.y[5]

jPos = sol.JupiterPos[0]
sPos = sol.SunPos[0]
    
# Energy per unit mass calculations

kineticEnergy = (1/2)*( (vx - w_jup*y)**2 + (vy + w_jup*x)**2 + vz**2)
    
potentialEnergy = -G*Msun / np.sqrt( (x - sPos[0])**2 + (y - sPos[1])**2 + (z - sPos[2])**2 ) - G*Mjup / np.sqrt( (x - jPos[0])**2 + (y - jPos[1])**2 + (z - jPos[2])**2 )
    
energyArray = kineticEnergy + potentialEnergy

print("Transform complete.")

print("Saving data...")

saveData = np.array([t, x, y, z, energyArray]).transpose()

np.savetxt("data.csv", saveData )

print("Saved.")



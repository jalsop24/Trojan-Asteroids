
import numpy as np
from solver import solver
import matplotlib.pyplot as plt

####################
# Solver Constants #
####################

method = "Radau" # Radau, DOP853 # RK45 Works with rotating frame but not inertial

T_Min = 0

T_Max = 500

Resolution = 500

inertial = False



######################
# Physical Constants #
######################

G = 4*np.pi**2 # Solar system units: unit time = 1 year, unit length = 1 AU



##########################
# Sun, Jupiter Constants #
##########################

alpha = 0.001

Msun = 1
Mjup = alpha * Msun

Rjup = 5.2 # Separation between Jupiter and the Sun

w_jup = np.sqrt( G*Msun/(Rjup**3) ) # Angular velocity of Jupiter around the CoM

theta = np.pi/6

point = np.array([ Rjup*np.cos(theta), Rjup*np.sin(theta), 0 ])

###############################
# Asteroid Initial Conditions #
###############################

asteroidAngle = np.pi/2 - np.pi/180 * 60

rMid= 5.2

rRange = 0.2
rStart = rMid - rRange/2

N = 40

wanderData = []
rData = []

for n in range(0, N):
    
    RAst = rStart + n/N * rRange
    
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

    t = sol.t
    
    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]
    
    vx = sol.y[3] 
    vy = sol.y[4]
    vz = sol.y[5]
    
    r_avg = np.array( [np.mean(x), np.mean(y), np.mean(z)] )
    
    r_relative = np.array([ np.abs(x - point[0]), np.abs(y - point[1]), np.abs(z - point[2]) ])
    
    mean_drift = r_avg - point
    
    
    
    rData.append(RAst)
    
    wanderData.append( np.sqrt( mean_drift[0]**2 + mean_drift[1]**2 + mean_drift[2]**2 ) )
    

plt.plot(rData, wanderData)

plt.xlabel("Radial Start Position / AU")
plt.ylabel("Asteroid Wander / AU")











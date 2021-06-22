# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:21:47 2020

@author: James
"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
#from CelestialBodyClass import CBody

######################
# Physical Constants #
######################

G = 4*np.pi**2

AU = 1

##########################
# Sun, Jupiter Constants #
##########################

alpha = 0

Msun = 1
Mjup = alpha * Msun

Rjup = 5.2 * AU

theta_jup = np.pi/2
x0_j = Rjup*np.cos(theta_jup)
y0_j = Rjup*np.sin(theta_jup)

###############################
# Asteroid Initial Conditions #
###############################

asteroidAngle = np.pi/180 * 30

RAst = 5.3 * AU

x0 = RAst * np.cos(asteroidAngle)
y0 = RAst * np.sin(asteroidAngle)

theta0 = 0 * np.pi/180 

orbitalSpeed = np.sqrt( G*Msun/(Rjup**3) ) * RAst * 1

vx0 = -orbitalSpeed*np.sin(asteroidAngle + theta0)
vy0 = orbitalSpeed*np.cos(asteroidAngle + theta0)

L = RAst * orbitalSpeed /2

####################
# Solver Constants #
####################

solver = "Radau"

T_Min = 0

T_Max = 5000

Resolution = 500

def sunEoM(t):
    
    return np.array([0, 0])

def jupEoM(t):
    
    r = Rjup
    
    w = np.sqrt( G*Msun/(Rjup**3) )
    
    return np.array([r*np.cos(theta_jup + w*t),  r*np.sin(theta_jup + w*t)])

def aField(t, vec):
    
    retVec = np.zeros(4)
    
    # Unpack asteroid position & velocity from the last step
    x = vec[0]
    y = vec[1]
    
    vx = vec[2]
    vy = vec[3]
    
    # Return the new position ODEs [ dx/dt = vx, dy/dt = vy ]
    retVec[0] = vx
    retVec[1] = vy
    
    # Get the current positions of the Sun and Jupiter
    sunPos = sunEoM(t)
    jupPos = jupEoM(t)
    
    # Asteroid parameters wrt the Sun
    Rs = np.sqrt( (x-sunPos[0])**2 + (y-sunPos[1])**2 )
    theta = np.arctan2(y-sunPos[1], x-sunPos[0])
    
    # Asteroid parameters wrt Jupiter
    jDist2 = (x-jupPos[0])**2 + (y-jupPos[1])**2
    phi = np.arctan2(y-jupPos[1], x-jupPos[0])
    
    # New velocity ODEs [ d(vx)/dt = ax , d(vy)/dt = ay ]
    retVec[2] = ( -G*Msun/Rs**2)*np.cos(theta) - (G*Mjup/jDist2)*np.cos(phi) # ax
    retVec[3] = ( -G*Msun/Rs**2)*np.sin(theta) - (G*Mjup/jDist2)*np.sin(phi) # ay
    
    return retVec

sol = solve_ivp(
    aField,
    [T_Min, round(T_Max, 0)],
    [x0, y0, vx0, vy0],
    dense_output=False,
    method=solver,
    t_eval=np.linspace(T_Min, round(T_Max, 0), round( (T_Max-T_Min)*Resolution) )
    )

# Transform the inertial co-ordinates into the rotating frame #

xArray = np.zeros(len(sol.t))
yArray = np.copy(xArray)

energyArray = np.copy(xArray)

for n in range(0, len(sol.t)):
    
    t = sol.t[n]
    
    x = sol.y[0][n]
    y = sol.y[1][n]
    
    vx = sol.y[2][n]
    vy = sol.y[3][n]
    
    sPos = sunEoM(t)
    
    jPos = jupEoM(t)
    
    theta = -np.sqrt( G*Msun/(Rjup**3) )*t 
    
    xArray[n] = np.cos(theta) * x - np.sin(theta) * y
    
    yArray[n] = np.sin(theta) * x + np.cos(theta) * y
    
    kineticEnergy = (1/2)*(vx**2 + vy**2)
    
    potentialEnergy = -G*Msun / np.sqrt( (x - sPos[0])**2 + (y - sPos[1])**2 ) - G*Mjup / np.sqrt( (x - jPos[0])**2 + (y - jPos[1])**2 )
    
    energyArray[n] = kineticEnergy + potentialEnergy


# plt.plot(sol.y[0], sol.y[1], label="Asteroid")

# plt.plot(jupEoM(sol.t)[0], jupEoM(sol.t)[1], label="Jupiter")

# plt.plot(sol.t, energyArray)

plt.plot(xArray, yArray, label="Relative Position")

plt.xlim(-10, 10)
plt.ylim(-10, 10)
    

plt.legend(loc="best")

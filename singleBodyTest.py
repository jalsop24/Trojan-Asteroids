# -*- coding: utf-8 -*-
"""
Created on Fri Apr 10 13:39:46 2020

@author: James
"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

####################
# Solver Constants #
####################

solver = "RK45" # Radau, DOP853 # RK45 Works with rotating frame but not inertial

T_Min = 0

T_Max = 500

Resolution = 1000

axesLimits = 8

inertial = True



######################
# Physical Constants #
######################

G = 4*np.pi**2 # Solar system units: unit time = 1 year, unit length = 1 AU



#################
# Sun Constants #
#################

Msun = 1

sunPos = [0,0,0]



###############################
# Asteroid Initial Conditions #
###############################

asteroidAngle = np.pi/180 * 30

RAst = 5.2

x0 = RAst * np.cos(asteroidAngle)
y0 = RAst * np.sin(asteroidAngle)
z0 = 0

theta0 = 0 * np.pi/180 

w = np.sqrt( G*Msun/(RAst**3) )

orbitalSpeed =  w * RAst 

vx0 = -orbitalSpeed*np.sin(asteroidAngle + theta0)
vy0 = orbitalSpeed*np.cos(asteroidAngle + theta0)
vz0 = 0

inertial_y0 = [x0, y0, z0, vx0, vy0, vz0]

rotating_y0 = [x0, y0, z0, vx0 + w*y0, vy0 - w*x0, vz0]



def inertialField(t, vec):
    
    retVec = np.zeros(6)
    
    # Unpack asteroid position & velocity from the last step
    x = vec[0]
    y = vec[1]
    z = vec[2]
    
    vx = vec[3]
    vy = vec[4]
    vz = vec[5]
    
    # Return the new position ODEs [ dr_i/dt = v_i ]
    retVec[0] = vx
    retVec[1] = vy
    retVec[2] = vz
    
    # Asteroid parameters wrt the Sun
    r_s     = np.sqrt( (x-sunPos[0])**2 + (y-sunPos[1])**2 + (z-sunPos[2])**2 )
    phi_s   = np.arctan2(y-sunPos[1], x-sunPos[0])
    theta_s = np.arccos( z/r_s )
    
    # New velocity ODEs [ d(v_i)/dt = F_i/m ]
    k = ( -G*Msun/r_s**2)
    retVec[3] = k*np.cos(phi_s)*np.sin(theta_s)  # ax
    retVec[4] = k*np.sin(phi_s)*np.sin(theta_s)  # ay
    retVec[5] = k*np.cos(theta_s)                # az
    
    return retVec

def rotatingField(t, vec):
    
    retVec = np.zeros(6)
    
    # Unpack asteroid position & velocity from the last step
    x = vec[0]
    y = vec[1]
    z = vec[2]
    
    vx = vec[3]
    vy = vec[4]
    vz = vec[5]
    
    # Return the new position ODEs [ dr_i/dt = v_i ]
    retVec[0] = vx
    retVec[1] = vy
    retVec[2] = vz
    
    # Asteroid parameters wrt the Sun
    r_s     = np.sqrt( (x-sunPos[0])**2 + (y-sunPos[1])**2 + (z-sunPos[2])**2 )
    phi_s   = np.arctan2(y-sunPos[1], x-sunPos[0])
    theta_s = np.arccos( z/r_s )
    
    # New velocity ODEs [ d(v_i)/dt = F_i/m ]
    k = ( -G*Msun/r_s**2)
    retVec[3] = k*np.cos(phi_s)*np.sin(theta_s) + w**2 * x + 2*w*vy   # ax
    retVec[4] = k*np.sin(phi_s)*np.sin(theta_s) + w**2 * y - 2*w*vx   # ay
    retVec[5] = k*np.cos(theta_s)                                     # az
    
    return retVec

sol = solve_ivp(
    inertialField if inertial else rotatingField,
    [T_Min, T_Max],
    inertial_y0 if inertial else rotating_y0,
    dense_output=False,
    vectorized=False,
    method=solver,
    t_eval=np.linspace(T_Min, T_Max, (T_Max - T_Min)*Resolution)
    )




# All operations and functions are in a vectorised form

t = sol.t
    
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]
    
vx = sol.y[3] 
vy = sol.y[4]
vz = sol.y[5]
 
theta = w*t if inertial else -w*t
    
xArray =  np.cos(theta) * x + np.sin(theta) * y
    
yArray = - np.sin(theta) * x + np.cos(theta) * y
    
kineticEnergy = (1/2)*(vx**2 + vy**2 + vz**2) if inertial else (1/2)*( (vx - w*y)**2 + (vy + w*x)**2 + vz**2)
    
potentialEnergy = -G*Msun / np.sqrt( (x - sunPos[0])**2 + (y - sunPos[1])**2 + (z - sunPos[2])**2 )
    
energyArray = kineticEnergy + potentialEnergy

plt.plot(sol.y[0], sol.y[1])

# plt.plot(xArray, yArray)

# plt.plot(xArray if inertial else sol.y[0], yArray if inertial else sol.y[1], label="Relative Position")

plt.xlim(-axesLimits, axesLimits)
plt.ylim(-axesLimits, axesLimits)

plt.xlabel("x / AU")
plt.ylabel("y / AU")






















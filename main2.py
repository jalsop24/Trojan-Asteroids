# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 14:21:47 2020

@author: James
"""


import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


####################
# Solver Constants #
####################

solver = "Radau" # Radau, DOP853 # RK45 Works with rotating frame but not inertial

T_Min = 0

T_Max = 100

Resolution = 1000

axesLimits = 8

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

theta_jup = np.pi/2

w_jup = np.sqrt( G*Msun/(Rjup**3) ) # Angular velocity of Jupiter around the CoM

r_j0 = Rjup/(1+alpha)

r_s0 = alpha * r_j0



###############################
# Asteroid Initial Conditions #
###############################

asteroidAngle = np.pi/180 * 30

RAst = 5.2

x0 = RAst * np.cos(asteroidAngle)
y0 = RAst * np.sin(asteroidAngle)
z0 = 0

theta0 = 0 * np.pi/180 

orbitalSpeed =  w_jup * RAst * 1

vx0 = -orbitalSpeed*np.sin(asteroidAngle + theta0)
vy0 = orbitalSpeed*np.cos(asteroidAngle + theta0)
vz0 = 0

inertial_y0 = [x0, y0, z0, vx0, vy0, vz0]

rotating_y0 = [x0, y0, z0, vx0 + w_jup*y0, vy0 - w_jup*x0, vz0]



def sunEoM(t, r=r_s0, w=w_jup, theta0=-theta_jup):
    
    return np.array([r*np.cos(theta0 + w*t),  r*np.sin(theta0 + w*t), 0])

def jupEoM(t, r=r_j0, w=w_jup, theta0=theta_jup):
    
    return np.array([r*np.cos(theta0 + w*t),  r*np.sin(theta0 + w*t), 0])


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
    
    # Get the current positions of the Sun and Jupiter
    sunPos = sunEoM(t)
    jupPos = jupEoM(t)
    
    # Asteroid parameters wrt the Sun
    r_s     = np.sqrt( (x-sunPos[0])**2 + (y-sunPos[1])**2 + (z-sunPos[2])**2 )
    phi_s   = np.arctan2(y-sunPos[1], x-sunPos[0])
    theta_s = np.arccos( z/r_s )
    
    # Asteroid parameters wrt Jupiter
    r_j     = np.sqrt( (x-jupPos[0])**2 + (y-jupPos[1])**2 + (z-jupPos[2])**2 )  
    phi_j   = np.arctan2(y-jupPos[1], x-jupPos[0])
    theta_j = np.arccos( z/r_j )
    
    # New velocity ODEs [ d(v_i)/dt = F_i/m ]
    retVec[3] = ( -G*Msun/r_s**2)*np.cos(phi_s)*np.sin(theta_s) - (G*Mjup/r_j**2)*np.cos(phi_j)*np.sin(theta_j) # ax
    retVec[4] = ( -G*Msun/r_s**2)*np.sin(phi_s)*np.sin(theta_s) - (G*Mjup/r_j**2)*np.sin(phi_j)*np.sin(theta_j) # ay
    retVec[5] = ( -G*Msun/r_s**2)*np.cos(theta_s)               - (G*Mjup/r_j**2)*np.cos(theta_j)               # az
    
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
    
    # Return the new position ODEs [ dx/dt = vx, dy/dt = vy ]
    retVec[0] = vx
    retVec[1] = vy
    retVec[2] = vz
    
    # Get the positions of the Sun and Jupiter (rotating frame keeps them held at constant positions)
    sunPos = sunEoM(0)
    jupPos = jupEoM(0)
    
    # Asteroid parameters wrt the Sun
    r_s     = np.sqrt( (x-sunPos[0])**2 + (y-sunPos[1])**2 + (z-sunPos[2])**2 )
    phi_s   = np.arctan2(y-sunPos[1], x-sunPos[0])
    theta_s = np.arccos( z/r_s )
    
    # Asteroid parameters wrt Jupiter
    r_j     = np.sqrt( (x-jupPos[0])**2 + (y-jupPos[1])**2 + (z-jupPos[2])**2 )  
    phi_j   = np.arctan2(y-jupPos[1], x-jupPos[0])
    theta_j = np.arccos( z/r_j )
    
    # New velocity ODEs [ d(vx)/dt = ax , d(vy)/dt = ay ]
    retVec[3] = ( -G*Msun/r_s**2)*np.cos(phi_s)*np.sin(theta_s) - (G*Mjup/r_j**2)*np.cos(phi_j)*np.sin(theta_j) + w_jup**2 * x + 2*w_jup*vy # ax
    retVec[4] = ( -G*Msun/r_s**2)*np.sin(phi_s)*np.sin(theta_s) - (G*Mjup/r_j**2)*np.sin(phi_j)*np.sin(theta_j) + w_jup**2 * y - 2*w_jup*vx # ay
    retVec[5] = ( -G*Msun/r_s**2)*np.cos(theta_s)               - (G*Mjup/r_j**2)*np.cos(theta_j)               # az
    
    return retVec




############################
# Run the solver algorithm #
############################


print("Starting solver...")

sol = solve_ivp(
    inertialField if inertial else rotatingField,
    [T_Min, round(T_Max, 0)],
    inertial_y0 if inertial else rotating_y0,
    method=solver,
    t_eval=np.linspace(T_Min, round(T_Max, 0), round( (T_Max-T_Min)*Resolution) ),
    )

print("Solver complete.")




###############################################################
# Transform the inertial coordinates into the rotating frame #
###############################################################

print("Transforming coordinates...")

# All operations and functions are in a vectorised form

t = sol.t
    
x = sol.y[0]
y = sol.y[1]
z = sol.y[2]
    
vx = sol.y[3] 
vy = sol.y[4]
vz = sol.y[5]
 
sPos = sunEoM(t)
    
jPos = jupEoM(t)
    
theta = w_jup*t if inertial else -w_jup
    
xArray =   np.cos(theta) * x + np.sin(theta) * y
    
yArray = - np.sin(theta) * x + np.cos(theta) * y
    
kineticEnergy = (1/2)*(vx**2 + vy**2 + vz**2) if inertial else (1/2)*( (vx - w_jup*y)**2 + (vy + w_jup*x)**2 + vz**2)
    
potentialEnergy = -G*Msun / np.sqrt( (x - sPos[0])**2 + (y - sPos[1])**2 + (z - sPos[2])**2 ) - G*Mjup / np.sqrt( (x - jPos[0])**2 + (y - jPos[1])**2 + (z - jPos[2])**2 )
    
energyArray = kineticEnergy + potentialEnergy

print("Transform complete.")

# print("Saving data...")

# np.savetxt("data.csv", np.array([sol.t, xArray, yArray, z]).transpose() )

# print("Saved.")

############
# Plotting #
############

fig = plt.figure()

plt.plot(sol.t, energyArray, label="Total Energy")

plt.ylim(-25, 4)

plt.xlabel("Time / Years")
plt.ylabel("Total Energy / Arbitrary Units")

# plt.plot(xArray if inertial else sol.y[0], yArray if inertial else sol.y[1], label="Relative Position")

# plt.xlabel("x / AU")
# plt.ylabel("y / AU")

# ax = fig.add_subplot(projection="3d", azim=-35, elev=30)
# ax.plot(xArray if inertial else sol.y[0], yArray if inertial else sol.y[1], sol.y[2], label="Relative Position")
# ax.set_zlim(-axesLimits, axesLimits)

# plt.xlim(-axesLimits, axesLimits)
# plt.ylim(-axesLimits, axesLimits)

plt.legend(loc="best")

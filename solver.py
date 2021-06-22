

import numpy as np
from scipy.integrate import solve_ivp


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


def circularEoM(t, r, w, theta0=-theta_jup):
    
    return np.array([r*np.cos(theta0 + w*t),  r*np.sin(theta0 + w*t), 0])


def inertialField(t, vec, Msun, Mjup, Rjup, r_s0, r_j0, w_jup):
    
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
    sunPos = circularEoM(t, r_s0, w_jup, theta0=-theta_jup)
    jupPos = circularEoM(t, r_j0, w_jup, theta0=theta_jup)
    
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


def rotatingField(t, vec, Msun, Mjup, Rjup, r_s0, r_j0, w_jup):
    
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
    
    # Get the positions of the Sun and Jupiter (rotating frame keeps them held at constant positions)
    sunPos = circularEoM(0, r_s0, w_jup, theta0=-theta_jup)
    jupPos = circularEoM(0, r_j0, w_jup, theta0=theta_jup)
    
    # Asteroid parameters wrt the Sun
    r_s     = np.sqrt( (x-sunPos[0])**2 + (y-sunPos[1])**2 + (z-sunPos[2])**2 )
    phi_s   = np.arctan2(y-sunPos[1], x-sunPos[0])
    theta_s = np.arccos( z/r_s )
    
    # Asteroid parameters wrt Jupiter
    r_j     = np.sqrt( (x-jupPos[0])**2 + (y-jupPos[1])**2 + (z-jupPos[2])**2 )  
    phi_j   = np.arctan2(y-jupPos[1], x-jupPos[0])
    theta_j = np.arccos( z/r_j )
    
    # New velocity ODEs [ d(v_i)/dt = F_i/m ]
    retVec[3] = ( -G*Msun/r_s**2)*np.cos(phi_s)*np.sin(theta_s) - (G*Mjup/r_j**2)*np.cos(phi_j)*np.sin(theta_j) + w_jup**2 * x + 2*w_jup*vy # ax
    retVec[4] = ( -G*Msun/r_s**2)*np.sin(phi_s)*np.sin(theta_s) - (G*Mjup/r_j**2)*np.sin(phi_j)*np.sin(theta_j) + w_jup**2 * y - 2*w_jup*vx # ay
    retVec[5] = ( -G*Msun/r_s**2)*np.cos(theta_s)               - (G*Mjup/r_j**2)*np.cos(theta_j)               # az
    
    return retVec


# The main solver class. 
# Takes initial conditions and returns the asteroid orbit solution in the rotating frame regardless of which field is being used

class solver():
    
    def __init__(self, initialConditions, method="Radau", T_Min=0, T_Max=5000, Resolution=1000, inertial=True, Msun=Msun, Mjup=Mjup, Rjup=Rjup):
        
        # Resolution variable determines how frequently the evaluation steps should be done per simulated year
        
        alpha = Mjup/Msun
    
        w_jup = np.sqrt( G*Msun/(Rjup**3) ) # Angular velocity of Jupiter around the CoM

        r_j0 = Rjup/(1+alpha)

        r_s0 = alpha * r_j0
        
        sol = solve_ivp(
            inertialField if inertial else rotatingField,
           [T_Min, round(T_Max, 0)],
           initialConditions,
           method=method,
           t_eval=np.linspace(T_Min, round(T_Max, 0), round( (T_Max-T_Min)*Resolution) ),
           args=(Msun, Mjup, Rjup, r_s0, r_j0, w_jup)
           )
    
        sunPos = circularEoM(sol.t, r_s0, w_jup, theta0=-theta_jup)
        jupPos = circularEoM(sol.t, r_j0, w_jup, theta0=theta_jup)
        
        if inertial:
            # Takes the solution that is given in the inertial frame and transforms it to the rotating frame
            
            theta = w_jup * sol.t
            
            x = sol.y[0]
            y = sol.y[1]
            z = sol.y[2]
    
            vx = sol.y[3] 
            vy = sol.y[4]
            vz = sol.y[5]
            
            rotatingY = [
                          np.cos(theta) * x + np.sin(theta) * y, # x
                        - np.sin(theta) * x + np.cos(theta) * y, # y
                        z,                                       # z
                        vx + w_jup*y, # vx
                        vy - w_jup*x, # vy
                        vz,           # vz
                        ]
            
            self.y = rotatingY
        else:
            self.y = sol.y
        
        self.t = sol.t
        self.SunPos = sunPos
        self.JupiterPos = jupPos

    

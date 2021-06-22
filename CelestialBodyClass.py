# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 11:07:43 2020

@author: James
"""


class CBody:
    
    def __init__(self, mass, position, velocity, eom=None):
        self.Mass = mass
        self.Position = position
        self.Velocity = velocity
        self.EoM = eom
    
    def SetEoM(self, eq):
        
        self.EoM = eq
    
    def GetPosition(self, t):
        
        return self.EoM(t)
        

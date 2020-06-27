# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 18:26:51 2018

@author: to_reilly
"""
import numpy as np
import scipy.ndimage as nd

mu = 1e-7

def magnetization(bRem, dimensions, shape = 'cube'):
    if shape == 'cube':
        dip_mom = bRem * dimensions**3 / (4*np.pi*mu)
    return dip_mom

def singleMagnet(position, dipoleMoment, simDimensions, resolution):
    
    #create mesh coordinates
    x = np.linspace(-simDimensions[0]/2 + position[0], simDimensions[0]/2 + position[0], int(simDimensions[0]*resolution+1), dtype=np.float32)
    y = np.linspace(-simDimensions[1]/2 + position[1], simDimensions[1]/2 + position[1], int(simDimensions[1]*resolution+1), dtype=np.float32)
    z = np.linspace(-simDimensions[2]/2 + position[2], simDimensions[2]/2 + position[2], int(simDimensions[2]*resolution+1), dtype=np.float32)
    x, y, z = np.meshgrid(x,y,z)
        
    vec_dot_dip = 3*(x*dipoleMoment[0] + y*dipoleMoment[1])
    
    #calculate the distance of each mesh point to magnet, optimised for speed
    #for improved memory performance move in to b0 calculations
    vec_mag = np.square(x) + np.square(y) + np.square(z)
    vec_mag_3 = np.power(vec_mag,1.5)
    vec_mag_5 = np.power(vec_mag,2.5)
    del vec_mag
    
    B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3), dtype=np.float32)

    #calculate contributions of magnet to total field, dipole always points in xy plane
    #so second term is zero for the z component
    B0[:,:,:,0] += np.divide(np.multiply(x, vec_dot_dip),vec_mag_5) - np.divide(dipoleMoment[0],vec_mag_3)
    B0[:,:,:,1] += np.divide(np.multiply(y, vec_dot_dip),vec_mag_5) - np.divide(dipoleMoment[1],vec_mag_3)
    B0[:,:,:,2] += np.divide(np.multiply(z, vec_dot_dip),vec_mag_5)
    
    return B0

def createHalbach(numMagnets = 24, rings = (-0.075,-0.025, 0.025, 0.075), radius = 0.145, magnetSize = 0.0254, kValue = 2, resolution = 1000, bRem = 1.3, simDimensions = (0.3, 0.3, 0.2)):
    
    #define vacuum permeability
    mu = 1e-7
    
    #positioning of the magnets in a circle
    angle_elements = np.linspace(0, 2*np.pi, numMagnets, endpoint=False)
    
    #Use the analytical expression for the z component of a cube magnet to estimate
    #dipole momentstrength for correct scaling. Dipole approximation only valid 
    #far-ish away from magnet, comparison made at 1 meter distance.

    dip_mom = magnetization(bRem, magnetSize)
    
    #create array to store field data
    B0 = np.zeros((int(simDimensions[0]*resolution)+1,int(simDimensions[1]*resolution)+1,int(simDimensions[2]*resolution)+1,3), dtype=np.float32)
    
    #create halbach array
    for row in rings:
        for angle in angle_elements:            
            position = (radius*np.cos(angle),radius*np.sin(angle), row)
            
            dip_vec = [dip_mom*np.cos(kValue*angle), dip_mom*np.sin(kValue*angle)]
            dip_vec = np.multiply(dip_vec,mu)
            
            #calculate contributions of magnet to total field, dipole always points in xy plane
            #so second term is zero for the z component
            B0 += singleMagnet(position, dip_vec, simDimensions, resolution)

    return B0

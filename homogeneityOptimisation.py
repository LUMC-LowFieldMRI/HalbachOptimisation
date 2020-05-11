# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 16:17:41 2018

@author: to_reilly
"""

import numpy as np
import halbachFields
import random
from deap import algorithms, base, tools,creator
import multiprocessing
import ctypes
import time
import matplotlib.pyplot as plt

    
def fieldError(shimVector):
    field = np.zeros(np.size(sharedShimMagnetsFields,0))
    for idx1 in range(0,np.size(shimVector)):
        field += sharedShimMagnetsFields[:,idx1,shimVector[idx1]]
    return (((np.max(field)-np.min(field))/np.mean(field))*1e6,)


if __name__ == "__main__":

    innerRingRadii = np.array([148, 151, 154, 156, 159, 162, 165, 168, 171, 174, 177, 180, 183, 186, 189, 192, 195, 198, 201])*1e-3
    innerNumMagnets = np.array([50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69])
    
    outerRingRadii = innerRingRadii + 21*1e-3
    outerNumMagnets = innerNumMagnets + 7
    
    resolution = 5
    numRings = 23
    ringSep = .022 
    magnetLength = (numRings - 1) * ringSep
    ringPositions = np.linspace(-magnetLength/2, magnetLength/2, numRings)
    
            
    #population
    popSim = 10000
    maxGeneration = 100
    
    DSV = 200*1e-3
    
###################################################################################################################
##########################################                               ##########################################
##########################################     Create spherical mask     ##########################################
##########################################                               ##########################################
###################################################################################################################
    
    simDimensions = (DSV,DSV,DSV)
        
    coordinateAxis = np.linspace(-simDimensions[0]/2,simDimensions[0]/2,int(1e3*simDimensions[0]/resolution + 1))
    coords = np.meshgrid(coordinateAxis, coordinateAxis, coordinateAxis)

    mask = np.zeros(np.shape(coords[0]))
    mask[np.square(coords[0]) + np.square(coords[1]) + np.square(coords[2]) <= (DSV/2)**2] = 1

    octantMask = np.copy(mask)
    octantMask[coords[0] < 0] = 0
    octantMask[coords[1] < 0] = 0
    octantMask[coords[2] < 0] = 0

    ringPositionsSymmetery = ringPositions[ringPositions >= 0]
    
    shimFields = np.zeros((int(np.sum(octantMask)), np.size(ringPositionsSymmetery), np.size(innerRingRadii)))
    
    for positionIdx, position in enumerate(ringPositionsSymmetery):
        for sizeIdx, ringSize in enumerate(innerRingRadii):
            if position == 0:
                rings = (0,)
            else:
                rings = (-position, position)
            fieldData = halbachFields.createHalbach(numMagnets = innerNumMagnets[sizeIdx], rings = rings, radius = innerRingRadii[sizeIdx], magnetSize = 0.012, resolution = 1e3/resolution, simDimensions = simDimensions)
            fieldData += halbachFields.createHalbach(numMagnets = outerNumMagnets[sizeIdx], rings = rings, radius = outerRingRadii[sizeIdx], magnetSize = 0.012, resolution = 1e3/resolution, simDimensions = simDimensions)
            shimFields[:,positionIdx, sizeIdx] = fieldData[octantMask == 1,0]
        
        
    sharedShimMagnetsFields_base = multiprocessing.Array(ctypes.c_double, np.size(shimFields))
    sharedShimMagnetsFields = np.ctypeslib.as_array(sharedShimMagnetsFields_base.get_obj())
    sharedShimMagnetsFields = sharedShimMagnetsFields.reshape(np.size(shimFields,0),np.size(shimFields,1),np.size(shimFields,2))
    sharedShimMagnetsFields[...] = shimFields[...]
    
    random.seed()
    
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    
    toolbox = base.Toolbox()

    toolbox.register("attr_bool", random.randint,0,np.size(sharedShimMagnetsFields,2)-1)

    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, np.size(sharedShimMagnetsFields, 1))
    
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    
    toolbox.register("evaluate", fieldError)
    
    toolbox.register("mate", tools.cxTwoPoint)
    
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=popSim)
    CXPB, MUTPB = 0.55, 0.4
    print("Start of evolution")
    startTime = time.time()
    fitnesses = list(map(toolbox.evaluate, pop))
    bestError = np.inf

    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    
    fits = [ind.fitness.values[0] for ind in pop]
        
    # Variable keeping track of the number of generations
    g = 0
    minTracker = np.zeros((maxGeneration))
    startEvolution = time.time()
    # Begin the evolution
    while g < maxGeneration:
        startTime = time.time()
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)
        
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
    
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):

            # cross two individuals with probability CXPB
            if random.random() < CXPB:
                toolbox.mate(child1, child2)

                # fitness values of the children
                # must be recalculated later
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:

            # mutate an individual with probability MUTPB
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
    
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        
        
        # The population is entirely replaced by the offspring
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        if min(fits) < bestError:
            #best in a generation is not per se best ever due to mutations, this tracks best ever
            print("BEST VECTOR: " + str(tools.selBest(pop, 1)[0]))
            bestError = min(fits)
            actualBestVector = tools.selBest(pop, 1)[0]
        
        minTracker[g-1]= min(fits)
        print("Evaluation took " + str(time.time()-startTime) + " seconds")        
        print("Minimum: %i ppm" % min(fits))
    
    print("-- End of (successful) evolution --")
    best_ind = tools.selBest(pop, 1)[0]
    print("Best individual is %s, %s" % (best_ind, best_ind.fitness.values))
            
    
    bestVector = np.array(actualBestVector)    
    
    shimmedField = np.zeros(np.shape(mask))
    for positionIdx, position in enumerate(ringPositionsSymmetery):
        if position == 0:
            rings = (0,)
        else:
            rings = (-position, position)
        shimmedField += halbachFields.createHalbach(numMagnets = innerNumMagnets[bestVector[positionIdx]], rings = rings, radius = innerRingRadii[bestVector[positionIdx]], magnetSize = 0.012, resolution = 1e3/resolution, simDimensions = simDimensions)[...,0]
        shimmedField += halbachFields.createHalbach(numMagnets = outerNumMagnets[bestVector[positionIdx]], rings = rings, radius = outerRingRadii[bestVector[positionIdx]], magnetSize = 0.012, resolution = 1e3/resolution, simDimensions = simDimensions)[...,0]

    mask[mask == 0] = np.nan
    
    maskedField = np.abs(np.multiply(shimmedField,mask))

    print("Shimmed mean: %.2f mT"%(1e3*np.nanmean(maskedField)))
    print("Shimmed homogeneity: %.4f mT" %(1e3*(np.nanmax(maskedField)-np.nanmin(maskedField))))
    print("Shimmed homogeneity: %i ppm" %(1e6*((np.nanmax(maskedField)-np.nanmin(maskedField))/np.nanmean(maskedField))))
    
    plt.figure()
    plt.plot(coordinateAxis*1e3, maskedField[int(np.floor(np.size(maskedField,0)/2)),int(np.floor(np.size(maskedField,1)/2)),:])
    plt.xlabel('X axis (mm)')
    plt.ylabel('Field strength (Tesla)')
    plt.legend()

    plt.figure()
    plt.semilogy(minTracker)
    plt.title("Min error Vs generations")
    plt.xlabel("Generation")
    plt.ylabel("Error")
    
    fig, ax  = plt.subplots(1,3)
    ax[0].imshow(maskedField[:,:,int(np.floor(np.size(maskedField,2)/2))])
    ax[1].imshow(maskedField[:,int(np.floor(np.size(maskedField,1)/2)),:])
    ax[2].imshow(maskedField[int(np.floor(np.size(maskedField,0)/2)),:,:])    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    


# SIRD_Model.py: Functions for using a SIRD feedback model to predict the spread of COVID-19
#by daniel march

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model

import SIRD_Model

#--------------------------------------------------------------------------------------------------------

def calcRecovered(I, D): #where I is total infections, not current infections
    R = np.zeros(len(I))
    R[13:] = I[:-13] + D[13:] #if infected are not dead by 13 days, assume recovery
    for i in range(len(I) - 13): 
    return R

def calcAsymptomatic(I): #assume any infected were asymptomatic 5 days ago, note that the last 5 days will be zeroed!
    A = np.zeros(len(I))
    A[:-5] = I[5:]
    return A

#for all functions assume nonLinVars = [q,b2,b3], q may or may not be used
#for all functions assume linVars = [b0, b1, kappa, gamma, nu] (kappa is asymptomatic to infected rate)
#let beta = b0 + b1/(1+(b2*I)**b3))

def getSAIRDFeedMatrix(nonLinVars, pop, A, I, R, D):
    q = nonLinVars[0]
    c0 = q*pop
    
    sirdMatrix = np.zeros((len(S) - 1, 5, 5))
    nextIterMatrix = np.zeros((len(S) - 1, 5, 1)) #the S(t+1), I(t+1), ... matrix
    
    #susceptible row, dS = 0
    sirdMatrix[:,0,0] = 0 #assume constant susceptiples, no change

    #asymptomatic row, dA = B(t)*(c0*I / c0 + I) - kA, B(t) = b0 + b1/(1+b2*I^b3)
    sirdMatrix[:,1,0] = (c0 * I[:-1]) / (c0 + I[:-1]) #b0
    sirdMatrix[:,1,1] = (c0 * I[:-1]) / (c0 + I[:-1]) * (1 / (1 + (nonLinVars[1]*I[:-1])**2)) #b1
    sirdMatrix[:,1,2] = -A #kappa
    
    #infected row
    sirdMatrix[:,2,2] = A #kappa
    sirdMatrix[:,2,3] = -I[:-1] #gamma
    sirdMatrix[:,2,4] = -I[:-1] #nu

    #recovered row
    sirdMatrix[:,3,3] = I[:-1] #gamma

    sirdMatrix[:,4,4] = I[:-1] #nu

    #populate the S(t+1), I(t+1), ... matrix
    nextIterMatrix[:,0,0] = 0 #no change
    nextIterMatrix[:,1,0] = A[1:] - A[:-1]
    nextIterMatrix[:,2,0] = I[1:] - I[:-1]
    nextIterMatrix[:,3,0] = R[1:] - R[:-1]
    nextIterMatrix[:,4,0] = D[1:] - D[:-1]

    return nextIterMatrix, sirdMatrix

def flattenSAIRDMatrix(y, X): #get the matrices in a 2d format, time dimension is put into the first
    T = len(y)
    rowCount = np.shape(y)[1]
    newY = np.zeros((T*rowCount, 1))
    newX = np.zeros((T*rowCount, np.shape(X)[2]))
    
    #map to flat matrix
    for t in range(T):
        for i in range(rowCount):
            newY[t*rowCount + i] = y[t, i]
            newX[t*rowCount + i] = X[t, i]
            
    return newY, newX

#--------------------------------------------------------------------------------------------------------


def errorSIRD(nonLinVars, linVars, pop, A, I, R, D, lamda, w): #the custom error function for SIRD    
    y, A = getSIRDFeedMatrix(nonLinVars, pop, A, I, R, D)
    
    totalError = 0
    #see paper for optimization function
    T = len(A)
    for t in range(T):
        totalError = totalError + (w**(T - t))*(np.linalg.norm((A[t] @ np.asarray(linVars)) - y[t].transpose(), ord=2)**2)
    
    #return (1.0/T) * np.linalg.norm((A @ params) - y.transpose(), ord=2)**2  + lamda*np.linalg.norm(params, ord=1)
    totalError = (1.0/T)*totalError #divide by timeframe
    totalError = totalError + lamda*np.linalg.norm(params, ord=1) #regularization error
    return totalError

def getLinVarsSAIRD(nonLinVars, pop, A, I, R, D, lamda, w): #calculate the linear vars for the SIRD model, b0, gamma, nu  
    y, X = getSIRDFeedMatrix(nonLinVars, pop, A, I, R, D)
    nextIterMatrix, sirdMatrix = flattenSIRDMatrix(y, X)
    
    rowCount = np.shape(y)[1]
    T = int(len(nextIterMatrix)/rowCount)
    
    #construct y and A, see paper for solving the lasso optimization
    y = np.zeros( (T*rowCount, 1) )
    X = np.zeros( (T*rowCount, np.shape(sirdMatrix)[1]) )
    
    for t in range(T):
        for i in range(rowCount):
            y[rowCount*t+i] = nextIterMatrix[rowCount*t+i] * np.sqrt(w**(T - t))
            X[rowCount*t+i] = sirdMatrix[rowCount*t+i] * np.sqrt(w**(T - t))
    
    try: #fit model using lasso
        model = linear_model.Lasso(alpha=lamda, fit_intercept=False, positive=True)
        model.fit(A,y)
        params = model.coef_
    except: #did not converge, set params to zero
        params = np.zeros((np.shape(A)[1]))
        #print("linal didn't converge")
    
    #totalError = (1.0/T) * np.linalg.norm((A @ params) - y.transpose(), ord=2)**2  + lamda*np.linalg.norm(params, ord=1)
    return list(params)

def gridNonLinVars(constraints, varResols, pop, A, I, R, D, lamda, w): #solve for non linear vars, q, b1, b2, b3
    
    #varSteps[:] = constraints[:][0] + (constraints[:][1] - constraints[:][0])/varResols[:]
    varSteps = []
    for i in range(len(constraints)):
        varSteps.append(constraints[i][0] + (constraints[i][1] - constraints[i][0])/varResols[i]) #min + (max - min)/resol
        if(varSteps[-1] == 0):
            varSteps[-1] = 1 #avoids infinite loop and zero step movement
            
    #note beta = b0/(1 + (b1*I + b2*D)^b3)
    #assume starting vals as best starting value
    #minVars = constraints[:][0]
    minVars = []
    for i in range(len(constraints)): #fill minVars with the minimum starting value
        minVars.append((constraints[i][0]))
    
    linVars = getLinVarsSAIRD(minVars, pop, A, I, R, D, lamda, w)
    minCost = errorSAIRD(minVars, paramArg, pop, A, I, R, D, lamda, w) #the custom error function for SIRD
    
    currVars = minVars.copy() #deep copy
    currCost = minCost
    varIndex = 0 #which var to iterate
    #while the var isn't above it's max
    continueLoop = True
    #this could be achieved by using many for loops, but this is a more generalized appraoch
    while(continueLoop):
    
        linVars = getLinVarsSIRD(currVars, pop, I, R, D, lamda, w)
        currCost = errorSIRD(currVars, paramArg, pop, I, R, D, lamda, w)
        if(currCost < minCost):
            minCost = currCost
            minVars = currVars.copy()
    
        #print("at: ", currVars, currCost)

        currVars[varIndex] = currVars[varIndex] + varSteps[varIndex]
        while(currVars[varIndex] > constraints[varIndex][1]): #move varIndex anditerate appropriately
                currVars[varIndex] = constraints[varIndex][0] #reset to minimum
                varIndex = varIndex + 1 #move to iterating the next variable

                if(varIndex == len(currVars)): #out of range, end Loop
                    continueLoop = False
                    break
                currVars[varIndex] = currVars[varIndex] + varSteps[varIndex] #iterate var        
        varIndex = 0 
       
    linVars = getLinVarsSIRD(minVars, pop, I, R, D, lamda, w) #set lin vars according to the min nonlin vars
    return minVars, linVars #return vars and linVars

def solveAllVars(nonLinConstraints, nonLinResol, pop, I, R, D, lamda, w):
    nonLinVars, linVars = gridNonLinVars(nonLinConstraints, nonLinResol, pop, I, R, D, lamda, w)

    print("Solution: ")
    print("q:  ", nonLinVars[0])
    print("a: ", nonLinVars[1])
    print("C: ", nonLinVars[2])
    print("b1: ", nonLinVars[3])
    print("b2: ", nonLinVars[4])
    print("b0: ", linVars[0])
    print("g:  ", linVars[1])
    print("nu: ", linVars[2])
    print("cost: ", errorSIRD(nonLinVars, linVars, pop, I, R, D, lamda, w))
    print() #spacer

    return nonLinVars, linVars

#------------------------------------------------------------------

def calculateBeta(nonLinVars, linVars, pop, I): #how to calculate beta as function of time
    
    q = nonLinVars[0]
    alpha = nonLinVars[1]
    C = nonLinVars[2]
    b0 = linVars[0]
    b1 = nonLinVars[-2]
    b2 = nonLinVars[-1]
    
    beta = np.zeros((len(I) - 1))
    for t in range(len(I) - 1):
        beta[t] = np.exp(-alpha*(t+C)) + (b0 / (1 + (b1*(I[t]/(pop*q)))**b2))
    
    return beta


#------------------------------------------------------------------

#predict the next some days using constant parameters, q and params will be calculated if not set, uses smoothing method  from paper
def calculateFeedFuture(nonLinVars, linVars, infect, recov, dead, pop, daysToPredict):
    
    #A=sirdmatrix, and dt=nextIterMatrix, if we know S(t) we should be able to predict S(t+1)
    q = nonLinVars[0]
    #set up matrices and starting info
    dt, A = getSIRDFeedMatrix(nonLinVars, pop, infect, recov, dead)
    suscept = q*pop - infect - recov - dead

    sirdPredict = np.zeros((len(A) + daysToPredict, 4, 3))
    dtPredict = np.zeros((len(dt) + daysToPredict, 4, 1))

    sirdPredict[0:len(A)] = A
    dtPredict[0:len(dt)] = dt

    susceptPredict = np.zeros(len(suscept) + daysToPredict)
    infectPredict = np.zeros(len(infect) + daysToPredict)
    recovPredict = np.zeros(len(recov) + daysToPredict)
    deadPredict = np.zeros(len(dead) + daysToPredict)

    susceptPredict[0:len(suscept)] = suscept
    infectPredict[0:len(infect)] = infect
    recovPredict[0:len(recov)] = recov
    deadPredict[0:len(dead)] = dead

    T = len(suscept) - 1
    for t in range(T, T + daysToPredict): #go from last element in known list to end of prediction, see paper for method
        
        temp1 = np.exp(-nonLinVars[1]*(t+nonLinVars[2])) # e^(-a(t+C)) * SI/S+I
        temp1 = temp1 * ((susceptPredict[t] * infectPredict[t]) / (susceptPredict[t] + infectPredict[t]))
        
        temp2 = (1 / (1 + (nonLinVars[-2]*(infectPredict[t]/(pop*q)))**nonLinVars[-1])) #(1 / (1+(b1I)^b2))
        
        #populate the 4x3 matrix with parameters
        sirdPredict[t,0,0] = -((susceptPredict[t] * infectPredict[t]) / (susceptPredict[t] + infectPredict[t])) * temp2
        sirdPredict[t,1,0] = ((susceptPredict[t] * infectPredict[t]) / (susceptPredict[t] + infectPredict[t])) * temp2
        sirdPredict[t,1,1] = -infectPredict[t]
        sirdPredict[t,1,2] = -infectPredict[t]
        sirdPredict[t,2,1] = infectPredict[t]
        sirdPredict[t,3,2] = infectPredict[t]

        dtPredict[t,:,0] = (sirdPredict[t] @ linVars)
        
        #print("to infect:", sirdPredict[t,1,0]*linVars[0])
        #print("to recov: ", (linVars[1] + linVars[2])*infectPredict[t])
        
        #find next SIRD, based on dtPredict[t] (which is S(t+1) - S(t)) to predict S(t) (and so on)
        susceptPredict[t+1] = susceptPredict[t] + dtPredict[t,0,0] + temp1
        infectPredict[t+1] = infectPredict[t] + dtPredict[t,1,0] - temp1
        recovPredict[t+1] = recovPredict[t] + dtPredict[t,2,0]
        deadPredict[t+1] = deadPredict[t] + dtPredict[t,3,0]
    
    return susceptPredict, infectPredict, recovPredict, deadPredict, q, linVars



#predict future days that are not known
def predictFeedFuture(nonLinVars, linVars, infect, recov, dead, pop, daysToPredict, graphVals=[True,True,True,True]):
    pS, pI, pR, pD, q, params = calculateFeedFuture(nonLinVars, linVars, infect, recov, dead, pop, daysToPredict)
    
    suscept = nonLinVars[0]*pop - infect - recov - dead
    
    #plot actual and predicted values
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(suscept, color='blue', label='suscpetible')
        ax.plot(pS, color='blue', label='suscpetible', linestyle='dashed')
    if(graphVals[1]):
        ax.plot(infect, color='orange', label='infected')
        ax.plot(pI, color='orange', label='infected', linestyle='dashed')
    if(graphVals[2]):
        ax.plot(recov, color='green', label='recovered')
        ax.plot(pR, color='green', label='recovered', linestyle='dashed')
    if(graphVals[3]):
        ax.plot(dead, color='black', label='dead')
        ax.plot(pD, color='black', label='dead', linestyle='dashed')
      
    #plot beta over time
    betaConst = SIRD_Model.calculateConstantParams(infect, recov, dead, pop, q)[0]
    betaConstGraph = np.ones((len(infect)-1))*betaConst #fill array with const value
    
    fig2, ax2 = plt.subplots(figsize=(18,8))
    ax2.plot(SIRD_Model.calculateAverageParams(infect, recov, dead, pop, q, graph=False)[:,0], color="red") #time varying beta
    ax2.plot(betaConstGraph, color="brown") #constant beta
    ax2.plot(calculateBeta(nonLinVars, linVars, pop, pI), color="orange")

    
#predict days that are known for testing purposes, predicts the end portion of the given data
def predictFeedMatch(nonLinVars, linVars, infect, recov, dead, pop, daysToPredict, graphVals=[True,True,True,True]):
    pS, pI, pR, pD, q, params = calculateFeedFuture(nonLinVars, linVars, infect[0:-daysToPredict], recov[0:-daysToPredict], dead[0:-daysToPredict], pop, daysToPredict)
    
    suscept = nonLinVars[0]*pop - infect - recov - dead
    
    #plot actual and predicted values
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(suscept, color='blue', label='suscpetible')
        ax.plot(pS, color='blue', label='suscpetible', linestyle='dashed')
    if(graphVals[1]):
        ax.plot(infect, color='orange', label='infected')
        ax.plot(pI, color='orange', label='infected', linestyle='dashed')
    if(graphVals[2]):
        ax.plot(recov, color='green', label='recovered')
        ax.plot(pR, color='green', label='recovered', linestyle='dashed')
    if(graphVals[3]):
        ax.plot(dead, color='black', label='dead')
        ax.plot(pD, color='black', label='dead', linestyle='dashed')
        
    #plot beta over time
    betaConst = SIRD_Model.calculateConstantParams(infect, recov, dead, pop, q)[0]
    betaConstGraph = np.ones((len(infect)-1))*betaConst #fill array with const value
    
    fig2, ax2 = plt.subplots(figsize=(18,8))
    ax2.plot(SIRD_Model.calculateAverageParams(infect, recov, dead, pop, q, graph=False)[:,0], color="red") #time varying beta
    ax2.plot(betaConstGraph, color="brown") #constant beta
    ax2.plot(calculateBeta(nonLinVars, linVars, pop, pI), color="orange")


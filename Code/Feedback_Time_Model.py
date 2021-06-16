# SIRD_Model.py: Functions for using a SIRD feedback model to predict the spread of COVID-19
#by daniel march

import numpy as np
import csv
import matplotlib.pyplot as plt
from sklearn import linear_model

import SIRD_Model

#--------------------------------------------------------------------------------------------------------

#for all functions assume nonLinVars = [q, alpha, C, b1, b2]
#for all functions assume linVars = [b0, gamma, nu]
#let beta = e^(-alhpa*(t+C)) + (b0 / (1 + (b1*I)**b2))

def getSIRDFeedMatrix(nonLinVars, pop, I, R, D):
    q = nonLinVars[0]
    S = q*pop - I - R - D
    
    sirdMatrix = np.zeros((len(S) - 1, 4, 3))
    nextIterMatrix = np.zeros((len(S) - 1, 4, 1)) #the S(t+1), I(t+1), ... matrix

    temp1 = np.zeros((len(S)-1))
    for t in range(len(S)-1):
        temp1[t] = np.exp(-nonLinVars[1]*(t+nonLinVars[2]))
    
    temp2 = (1 / (1 + (nonLinVars[-2]*(I[:-1]/(pop*q)))**nonLinVars[-1]))
    
    sirdMatrix[:,0,0] = -(S[:-1] * I[:-1]) / (S[:-1] + I[:-1]) * temp2

    sirdMatrix[:,1,0] = (S[:-1] * I[:-1]) / (S[:-1] + I[:-1]) * temp2
    sirdMatrix[:,1,1] = -I[:-1]
    sirdMatrix[:,1,2] = -I[:-1]

    sirdMatrix[:,2,1] = I[:-1]

    sirdMatrix[:,3,2] = I[:-1]

    #populate the S(t+1), I(t+1), ... matrix
    nextIterMatrix[:,0,0] = S[1:] - S[:-1] - ((S[:-1] * I[:-1]) / (S[:-1] + I[:-1]) * temp1[:])
    nextIterMatrix[:,1,0] = I[1:] - I[:-1] + ((S[:-1] * I[:-1]) / (S[:-1] + I[:-1]) * temp1[:])
    nextIterMatrix[:,2,0] = R[1:] - R[:-1]
    nextIterMatrix[:,3,0] = D[1:] - D[:-1]

    return nextIterMatrix, sirdMatrix

def flattenSIRDMatrix(y, A): #get the matrices in a 2d format, time dimension is put into the first
    T = len(y)
    newY = np.zeros((T*4, 1))
    newA = np.zeros((T*4, 3))
    
    for t in range(T):
        newY[t*4 + 0] = y[t, 0]
        newY[t*4 + 1] = y[t, 1]
        newY[t*4 + 2] = y[t, 2]
        newY[t*4 + 3] = y[t, 3]
        
        newA[t*4 + 0] = A[t, 0]
        newA[t*4 + 1] = A[t, 1]
        newA[t*4 + 2] = A[t, 2]
        newA[t*4 + 3] = A[t, 3]
    return newY, newA

#--------------------------------------------------------------------------------------------------------


def errorSIRD(nonLinVars, linVars, pop, I, R, D, lamda, w): #the custom error function for SIRD    
    y, A = getSIRDFeedMatrix(nonLinVars, pop, I, R, D)
    
    totalError = 0
    #see paper for optimization function
    T = len(A)
    for t in range(T):
        totalError = totalError + (w**(T - t))*(np.linalg.norm((A[t] @ np.asarray(linVars)) - y[t].transpose(), ord=2)**2)
    
    #return (1.0/T) * np.linalg.norm((A @ params) - y.transpose(), ord=2)**2  + lamda*np.linalg.norm(params, ord=1)
    totalError = (1.0/T)*totalError #divide by timeframe
    totalError = totalError + lamda*np.linalg.norm(params, ord=1) #regularization error
    return totalError

def getLinVarsSIRD(nonLinVars, pop, I, R, D, lamda, w): #calculate the linear vars for the SIRD model, b0, gamma, nu  
    y, A = getSIRDFeedMatrix(nonLinVars, pop, I, R, D)
    nextIterMatrix, sirdMatrix = flattenSIRDMatrix(y, A)
    
    T = int(len(nextIterMatrix)/4)
    
    #construct y and A, see paper for solving the lasso optimization
    y = np.zeros( (T*4, 1) )
    A = np.zeros( (T*4, np.shape(sirdMatrix)[1]) )
    
    for t in range(T):
        
        y[4*t+0] = nextIterMatrix[4*t+0] * np.sqrt(w**(T - t))
        y[4*t+1] = nextIterMatrix[4*t+1] * np.sqrt(w**(T - t))
        y[4*t+2] = nextIterMatrix[4*t+2] * np.sqrt(w**(T - t))
        y[4*t+3] = nextIterMatrix[4*t+3] * np.sqrt(w**(T - t))
        
        A[4*t+0] = sirdMatrix[4*t+0] * np.sqrt(w**(T - t))
        A[4*t+1] = sirdMatrix[4*t+1] * np.sqrt(w**(T - t))
        A[4*t+2] = sirdMatrix[4*t+2] * np.sqrt(w**(T - t))
        A[4*t+3] = sirdMatrix[4*t+3] * np.sqrt(w**(T - t))
    
    try: #fit model using lasso
        model = linear_model.Lasso(alpha=lamda, fit_intercept=False, positive=True)
        model.fit(A,y)
        params = model.coef_
        
    except: #did not converge, set params to zero
        params = np.zeros((np.shape(A)[1]))
        #print("linal didn't converge")
    
    #totalError = (1.0/T) * np.linalg.norm((A @ params) - y.transpose(), ord=2)**2  + lamda*np.linalg.norm(params, ord=1)
    return list(params)

def gridNonLinVars(constraints, varResols, pop, I, R, D, lamda, w): #solve for non linear vars, q, b1, b2, b3
    
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
    
    linVars = getLinVarsSIRD(minVars, pop, I, R, D, lamda, w)
    minCost = errorSIRD(minVars, paramArg, pop, I, R, D, lamda, w) #the custom error function for SIRD
    
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
        
        print("to infect:", sirdPredict[t,1,0]*linVars[0])
        print("to recov: ", (linVars[1] + linVars[2])*infectPredict[t])
        
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


import numpy as np
import matplotlib.pyplot as plt

#this model is for the feedback version of SIRD, but assumes the susceptible population is always the same
# S' = 0
# I' = beta * (SI/S+I) - gamma*I - nu*I
# R' = gamma*I
# D' = nu*I

#where beta = b0 + b1/(1 + (b2*I) ** b3 #note that I is actually I / q*pop or I/(S+I+R+D)
#linVars = [beta0, beta1, gamma, nu]
#nonLinVars = [beta2, beta3]

#--------------------------------------------------------------------------
#global variables used for many functions
regularizer = 0
weightDecay = 1
betaUseDecay = False

delay = 21
S = 0 #the number of susceptibles, always stays constant

def setSuscept(q, pop):
    S = q*pop #set global var

#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
#create the basic model matrices
def getMatrix(I, R, D, nonLinVars):    
    sirdMatrix = np.zeros((len(I) - 1, 4, 4))
    nextIterMatrix = np.zeros((len(I) - 1, 4, 1)) #the S(t+1), I(t+1), ... matrix
    
    shiftI = np.zeros(len(I))
    shiftI[delay:] = I[:-delay]
    
    #susceptible row, dS = 0

    #infected row
    sirdMatrix[:,1,0] = (S * I[:-1]) / (S + I[:-1]) #beta0
    sirdMatrix[:,1,1] = (S * I[:-1]) / (S + I[:-1]) * (1 / (1 + (nonLinVars[0]*shiftI[:-1]/S)**nonLinVars[1] )) #beta1
    sirdMatrix[:,1,2] = -I[:-1] #gamma
    sirdMatrix[:,1,3] = -I[:-1] #nu

    #recovered row
    sirdMatrix[:,2,2] = I[:-1] #gamma

    #dead row
    sirdMatrix[:,3,3] = I[:-1] #nu

    #populate the S(t+1), I(t+1), ... matrix
    nextIterMatrix[:,0,0] = 0 #no change
    nextIterMatrix[:,1,0] = I[1:] - I[:-1]
    nextIterMatrix[:,2,0] = R[1:] - R[:-1]
    nextIterMatrix[:,3,0] = D[1:] - D[:-1]

    return nextIterMatrix, sirdMatrix
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
#solve for parameters using weight decay and solving row by row
def getLinVars(I, R, D, nonLinVars, graph=False):
    nu = getNu(I,D)
    gamma = getGamma(I,R)
    beta0, beta1 = getBeta(I,R,D, nonLinVars, gamma,nu)
    
    linVars = [beta0, beta1, gamma, nu]
    
    if(graph):
        fig, ax = plt.subplots(figsize=(18,8))
        ax.plot(getBetaTime(I,R,D, linVars, nonLinVars))
        return linVars, fig, ax
    
    return linVars


def getGamma(I, R):
    y = np.zeros((len(I)-1,1))
    x = np.zeros((len(I)-1,1))
    
    # R(t+1) - R(t) = gamma*I(t)
    y[:,0] = R[1:] - R[:-1]
    x[:,0] = I[:-1]

    #add weight decay
    T = len(I)-1
    for t in range(T):
        y[t] = y[t] * np.sqrt(weightDecay**(T-t))
        x[t] = x[t] * np.sqrt(weightDecay**(T-t))
    
    return np.linalg.lstsq(x, y, rcond = None)[0].flatten()[0] #solve for gamma

def getNu(I, D):
    y = np.zeros((len(I)-1,1))
    x = np.zeros((len(I)-1,1))
    
    # D(t+1) - D(t) = nu*I(t)
    y[:,0] = D[1:] - D[:-1]
    x[:,0] = I[:-1]

    #add weight decay
    T = len(I)-1
    for t in range(T):
        y[t] = y[t] * np.sqrt(weightDecay**(T-t))
        x[t] = x[t] * np.sqrt(weightDecay**(T-t))
    
    return np.linalg.lstsq(x, y, rcond = None)[0].flatten()[0] #solve for nu

def getBeta(I,R,D , nonLinVars, gamma, nu):
    y = np.zeros((len(I)-1,1))
    x = np.zeros((len(I)-1,2))
    
    #betaNonLin = [b2,b3]
    #dI = beta * (SI/S+I)
    
    shiftI = np.zeros(len(I))
    shiftI[delay:] = I[:-delay]
    
    #I'
    y[:,0] = (I[1:] - I[:-1]) + gamma*I[:-1] + nu*I[:-1]
    
    x[:,0] = (S*I[:-1]) / (S + I[:-1]) #beta0 
    x[:,1] = (S*I[:-1]) / (S + I[:-1]) * (1 / (1 + (nonLinVars[0]*shiftI[:-1]/S)**nonLinVars[1] ))
    
    if(betaUseDecay):
        #add weight decay
        T = len(I)-1
        for t in range(T):
            y[t] = y[t] * np.sqrt(weightDecay**(T-t))
            x[t] = x[t] * np.sqrt(weightDecay**(T-t))
    
    betaVars = np.linalg.lstsq(x, y, rcond = None)[0].flatten() #solve for beta
    return betaVars[0], betaVars[1] #beta0, beta1
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------
#find the error of the current parameters
def getError(I, R, D, linVars, nonLinVars, regError=True): #the custom error function for SIRD    
    y, x = getMatrix(I, R, D, nonLinVars)
    
    totalError = 0
    #see paper for optimization function
    T = len(y)
    for t in range(T):
        #add weight decay
        y[t] = y[t] * np.sqrt(weightDecay**(T-t))
        for row in range(len(x[t])):
            x[t,row] = x[t,row] * np.sqrt(weightDecay**(T-t))

        totalError = totalError + (np.linalg.norm((x[t] @ np.asarray(linVars)) - y[t].transpose(), ord=2)**2)
 
    #return (1.0/T) * np.linalg.norm((A @ params) - y.transpose(), ord=2)**2  + lamda*np.linalg.norm(params, ord=1)
    totalError = (1.0/T)*totalError #divide by timeframe
    if(regError):
        totalError = totalError + regularizer*np.linalg.norm(linVars, ord=1) #regularization error
    
    return totalError
#--------------------------------------------------------------------------

def getBetaTime(I, R, D, linVars, nonLinVars): #calculate beta from b0, b1, b2, b3
    #beta = b0 + b1/(1+b2*I**b3)
    
    shiftI = np.zeros(len(I))
    shiftI[delay:] = I[:-delay]
    
    return (linVars[0] + (linVars[1] / (1 + (nonLinVars[0] * shiftI/S)**nonLinVars[1] ) )) #beta over time, I/pop for normalization
    


#--------------------------------------------------------------------------
#solve for nonlinear variables [beta2, beta3]

#constraints should be in the format of [(b2min, b2max), (b3min b3max)]
def gridNonLinVars(I,R,D, constraints, varResols): #solve for non linear vars, q, b1, b2, b3
    
    #get the step distance needed for each variable
    #varSteps[:] = constraints[:][0] + (constraints[:][1] - constraints[:][0])/varResols[:]
    varSteps = []
    for i in range(len(constraints)):
        varSteps.append(constraints[i][0] + (constraints[i][1] - constraints[i][0])/varResols[i]) #min + (max - min)/resol
        if(varSteps[-1] == 0):
            varSteps[-1] = 1 #avoids infinite loop and zero step movement
            

    #let starting vals be known as best starting value
    #minVars = constraints[:][0]
    minVars = []
    for i in range(len(constraints)): #fill minVars with the minimum starting value
        minVars.append((constraints[i][0]))
    
    linVars = getLinVars(I, R, D, minVars)
    minCost = getError(I, R, D, linVars, minVars) #the custom error function for SIRD
    
    currVars = minVars.copy() #deep copy
    currCost = minCost
    #while the var isn't above it's max
    continueLoop = True
    #this could be achieved by using many for loops, but this is a more generalized appraoch
    while(continueLoop):
        linVars = getLinVars(I, R, D, currVars)
        currCost = getError(I, R, D, linVars, currVars) #error function for SIRD
        if(currCost < minCost):
            minCost = currCost
            minVars = currVars.copy()
    
        #handle iteration of variables without overflowing
        currVars[0] = currVars[0] + varSteps[0]
        varIndex = 0
        while(currVars[varIndex] > constraints[varIndex][1]): #move varIndex and iterate appropriately
                currVars[varIndex] = constraints[varIndex][0] #reset to minimum
                varIndex = varIndex + 1 #move to iterating the next variable

                if(varIndex == len(currVars)): #out of range, end Loop
                    continueLoop = False
                    break
                currVars[varIndex] = currVars[varIndex] + varSteps[varIndex] #iterate var        
       
    linVars = getLinVars(I, R, D, minVars) #set lin vars according to the min nonlin vars
    return linVars, minVars #return vars and linVars

def solveAllVars(I, R, D, nonLinConstraints, nonLinResol, printOut=False):
    linVars, nonLinVars = gridNonLinVars(I,R,D, nonLinConstraints, nonLinResol)

    if(printOut):
        print("Solution: ")
        print("b0: ", linVars[0])
        print("b1: ", linVars[1])
        print("g:  ", linVars[2])
        print("nu: ", linVars[3])
        print("b2: ", nonLinVars[0])
        print("b3: ", nonLinVars[1])
        print("cost: ", getError(I,R,D, linVars, nonLinVars) )
        print() #spacer

    return linVars, nonLinVars


#--------------------------------------------------------------------------
    
    
    
    
#--------------------------------------------------------------------------  
def displayData(I,R,D, graphVals=[True,True,True]):
    
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(I, color="red")
    if(graphVals[1]):
        ax.plot(R, color="blue")
    if(graphVals[2]):
        ax.plot(D, color="black")
        
    return fig, ax
#--------------------------------------------------------------------------  





#------------------------------------------------------------------
#prediction functions
#predict the next some days using constant parameters, q and params will be calculated if not set, uses smoothing method  from paper
def calculateFuture(I,R,D, daysToPredict, params=None, nonLinParams=None):
    if(nonLinParams==None): #nonlinParams not calculate them
        varRanges = [(0,5000), (0,10)]
        varResol = [100, 10]
        params, nonLinParams = solveAllVars(I,R,D, varRanges, varResol)
    
    if(params==None): #nonLinParams are set but params aren't
        params=getLinVars(I,R,D, nonLinVars)
        
    print("Non Lin Vars:", nonLinParams)
    print("Lin Vars:", params)
    
    #set up matrices and starting info
    dt, X = getMatrix(I,R,D, nonLinParams)

    xPredict = np.zeros((len(X) + daysToPredict, np.shape(X)[1], np.shape(X)[2]))
    dtPredict = np.zeros((len(dt) + daysToPredict, np.shape(dt)[1], 1))

    xPredict[0:len(X)] = X
    dtPredict[0:len(dt)] = dt
    
    IP = np.zeros(len(I) + daysToPredict)
    RP = np.zeros(len(R) + daysToPredict)
    DP = np.zeros(len(D) + daysToPredict)

    IP[0:len(I)] = I
    RP[0:len(R)] = R
    DP[0:len(D)] = D

    
    T = len(I) - 1
    for t in range(T, T + daysToPredict): #go from last element in known list to end of prediction, see paper for method
        
        shiftI = IP[t-delay] #the infected number shifted days ago
        
        #populate the 5x5 matrix with parameters
        #susceptible row, dS = 0 (no change)
        xPredict[t,0,0] = 0

        #infected row, dA = B*(S*I / S + I)
        xPredict[t,1,0] = (S * IP[t]) / (S + IP[t]) #b0
        xPredict[t,1,1] = (S * IP[t]) / (S + IP[t]) * (1 / (1 + (nonLinParams[0]*shiftI/S)**nonLinParams[1] )) #b1
        xPredict[t,1,2] = -IP[t] #gamma
        xPredict[t,1,3] = -IP[t] #nu

        #recovered row
        xPredict[t,2,2] = IP[t] #gamma

        #dead row
        xPredict[t,3,3] = IP[t] #nu

        #predict next iter matrix
        dtPredict[t,:,0] = (xPredict[t] @ params) 
        
        #find next SIRD, based on dtPredict[t] (which is S(t+1) - S(t)) to predict S(t) (and so on)
        IP[t+1] = IP[t] + dtPredict[t,1,0]
        RP[t+1] = RP[t] + dtPredict[t,2,0]
        DP[t+1] = DP[t] + dtPredict[t,3,0]
    
    return IP, RP, DP



#predict future days that are not known
def predictFuture(I,R,D, daysToPredict, linVars=None, nonLinVars=None, graphVals=[False,True,True,True]):
    pI, pR, pD = calculateFuture(I,R,D, daysToPredict, params=linVars, nonLinParams=nonLinVars)

    #plot actual and predicted values
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(I, color='red', label='infected')
        ax.plot(pI, color='red', label='infected', linestyle='dashed')
    if(graphVals[1]):
        ax.plot(R, color='blue', label='recovered')
        ax.plot(pR, color='blue', label='recovered', linestyle='dashed')
    if(graphVals[2]):
        ax.plot(D, color='black', label='dead')
        ax.plot(pD, color='black', label='dead', linestyle='dashed')
      
    return pI, pR, pD, fig, ax #for easy manipulation/graphing

    
#predict days that are known for testing purposes, predicts the end portion of the given data
def predictMatch(I,R,D, daysToPredict, linVars=None, nonLinVars=None, graphVals=[False,True,True,True]):
    pI, pR, pD = calculateFuture(I[0:-daysToPredict], R[0:-daysToPredict], D[0:-daysToPredict], daysToPredict, params=linVars, nonLinParams=nonLinVars)
    
    #plot actual and predicted values
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(I, color='red', label='infected')
        ax.plot(pI, color='red', label='infected', linestyle='dashed')
    if(graphVals[1]):
        ax.plot(R, color='blue', label='recovered')
        ax.plot(pR, color='blue', label='recovered', linestyle='dashed')
    if(graphVals[2]):
        ax.plot(D, color='black', label='dead')
        ax.plot(pD, color='black', label='dead', linestyle='dashed')
      
    return pI, pR, pD, fig, ax #for easy manipulation/graphing

#---------------------------------------------------------





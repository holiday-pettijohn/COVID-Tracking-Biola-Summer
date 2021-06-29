import numpy as np
import matplotlib.pyplot as plt

#this model is for the feedback version of SIRD
# S' = -beta * (SA/S+A)
# A' = beta * (SA/S+A) - k*A
# I' = kappa*A - gamma*I - nu*I
# R' = gamma*I
# D' = nu*I

#where beta = b0 + b1/(1 + (b2*I) ** b3 #note that I is actually I / q*pop or I/(S+I+R+D)
#linVars = [beta0, beta1, gamma, nu]
#nonLinVars = [beta2, beta3]

#--------------------------------------------------------------------------
#global variables used for many functions
regularizer = 0
weightDecay = 1
betaUseDecay = False #since beta is modeled with feedback, by default it doesn't use weightDecay
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
#create the basic model matrices
def getMatrix(S, A, I, R, D, nonLinVars):    
    sirdMatrix = np.zeros((len(S) - 1, 5, 5)) #beta0, beta1, kappa, gamma, nu
    nextIterMatrix = np.zeros((len(S) - 1, 5, 1)) #the S(t+1), I(t+1), ... matrix
    
    pop = S+A+I+R+D #for normalizing I in feedback
    
    newI = (I+R+D)[1:] - (I+R+D)[:-1] #use new infections for the feedback
    
    #susceptible row, dS = 0
    sirdMatrix[:,0,0] = -(S[:-1] * A[:-1]) / (S[:-1] + A[:-1]) #beta0
    sirdMatrix[:,0,1] = -(S[:-1] * A[:-1]) / (S[:-1] + A[:-1]) * (1 / (1 + (nonLinVars[0]*newI/pop[:-1])**nonLinVars[1] )) #beta1

    #asymptomatic row, dA = 0
    sirdMatrix[:,1,0] = (S[:-1] * A[:-1]) / (S[:-1] + A[:-1]) #beta0
    sirdMatrix[:,1,1] = (S[:-1] * A[:-1]) / (S[:-1] + A[:-1]) * (1 / (1 + (nonLinVars[0]*newI/pop[:-1])**nonLinVars[1] )) #beta1
    sirdMatrix[:,1,2] = -A[:-1] #kappa
    
    #infected row
    sirdMatrix[:,2,2] = A[:-1] #kappa
    sirdMatrix[:,2,3] = -I[:-1] #gamma
    sirdMatrix[:,2,4] = -I[:-1] #nu

    #recovered row
    sirdMatrix[:,3,3] = I[:-1] #gamma

    #dead row
    sirdMatrix[:,4,4] = I[:-1] #nu

    #populate the S(t+1), I(t+1), ... matrix
    nextIterMatrix[:,0,0] = S[1:] - S[:-1]
    nextIterMatrix[:,1,0] = A[1:] - A[:-1]
    nextIterMatrix[:,2,0] = I[1:] - I[:-1]
    nextIterMatrix[:,3,0] = R[1:] - R[:-1]
    nextIterMatrix[:,4,0] = D[1:] - D[:-1]

    return nextIterMatrix, sirdMatrix
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
#solve for parameters using weight decay and solving row by row
def getLinVars(S, A, I, R, D, nonLinVars, graph=False):
    nu = getNu(I,D)
    gamma = getGamma(I,R)
    kappa = getKappa(A,I, gamma, nu)
    
    beta0, beta1 = getBeta(S,A,I,R,D, nonLinVars, kappa)
    
    linVars = [beta0, beta1, kappa, gamma, nu]
    
    if(graph):
        fig, ax = plt.subplots(figsize=(18,8))
        ax.plot(getBetaTime(S,A,I,R,D, linVars, nonLinVars))
        return linVars, fig, ax
    
    return linVars


def getKappa(A, I, gamma, nu):
    y = np.zeros((len(I)-1,1))
    x = np.zeros((len(I)-1,1))
    
    #I' = kappa*A - gamma*I - nu*I
    #I' + gamma*I + nu*I = kappa*A
    y[:,0] = A[1:] - A[:-1] + gamma*I[:-1] + nu*I[:-1]
    x[:,0] = A[:-1]
    
    #add weight decay
    T = len(I)-1
    for t in range(T):
        y[t] = y[t] * np.sqrt(weightDecay**(T-t))
        x[t] = x[t] * np.sqrt(weightDecay**(T-t))

    return np.linalg.lstsq(x, y, rcond = None)[0].flatten()[0] #solve for gamma

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

def getBeta(S,A,I,R,D , nonLinVars, kappa):
    y = np.zeros((2*(len(I)-1),1)) # 2 times length since every other row is for S' and every other is I'
    x = np.zeros((2*(len(I)-1),2))
    
    newI = (I+R+D)[1:] - (I+R+D)[:-1] #use new infections for the feedback
    
    #betaNonLin = [b2,b3]
    #dS = -beta * (SI/S+I)
    #dA = beta * (SI/S+I) - kappa*A
    
    pop = S+A+I+R+D #for normalizing b1*I
    
    #S and I rows
    y[::2,  0] = (S[1:] - S[:-1]) #::2 is for skipping every other row (starts at 0)
    y[1::2, 0] = (A[1:] - A[:-1]) + kappa*I[:-1]
    
    x[::2,  0] = -(S[:-1]*A[:-1]) / (S[:-1] + A[:-1]) #beta0
    x[1::2, 0] = (S[:-1]*A[:-1]) / (S[:-1] + A[:-1]) #beta0 
    
    x[::2,  1] = -(S[:-1]*A[:-1]) / (S[:-1] + A[:-1]) * (1 / (1 + (nonLinVars[0]*newI/pop[:-1])**nonLinVars[1] )) #beta1
    x[1::2, 1] = (S[:-1]*A[:-1]) / (S[:-1] + A[:-1]) * (1 / (1 + (nonLinVars[0]*newI/pop[:-1])**nonLinVars[1] )) #beta1
    
    if(betaUseDecay):
        #add weight decay
        T = len(I)-1
        for t in range(T):
            x[t*2:(t*2)+2] = x[t*2:(t*2)+2] * np.sqrt(weightDecay**(T - t))
            y[t*2:(t*2)+2] = y[t*2:(t*2)+2] * np.sqrt(weightDecay**(T - t))
    
    betaVars = np.linalg.lstsq(x, y, rcond = None)[0].flatten() #solve for beta
    return betaVars[0], betaVars[1] #beta0, beta1
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------
#find the error of the current parameters
def getError(S, A, I, R, D, linVars, nonLinVars, regError=True): #the custom error function for SIRD    
    y, x = getMatrix(S, A, I, R, D, nonLinVars)
    
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

def getBetaTime(S, A, I, R, D, linVars, nonLinVars): #calculate beta from b0, b1, b2, b3
    #beta = b0 + b1/(1+b2*I**b3)
    pop = (S+A+I+R+D)[0] #pop should be the same no matter what
    
    newI = (I+R+D)[1:] - (I+R+D)[:-1]
    return (linVars[0] + (linVars[1] / (1 + (nonLinVars[0] * newI/pop)**nonLinVars[1] ) )) #beta over time, I/pop for normalization
    


#--------------------------------------------------------------------------
#solve for nonlinear variables [beta2, beta3]

#constraints should be in the format of [(b2min, b2max), (b3min b3max)]
def gridNonLinVars(S,A,I,R,D, constraints, varResols): #solve for non linear vars, q, b1, b2, b3
    
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
    
    linVars = getLinVars(S, A, I, R, D, minVars)
    minCost = getError(S, A, I, R, D, linVars, minVars) #the custom error function for SIRD
    
    currVars = minVars.copy() #deep copy
    currCost = minCost
    #while the var isn't above it's max
    continueLoop = True
    #this could be achieved by using many for loops, but this is a more generalized appraoch
    while(continueLoop):
        linVars = getLinVars(S, A, I, R, D, currVars)
        currCost = getError(S, A, I, R, D, linVars, currVars) #error function for SIRD
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
       
    linVars = getLinVars(S, A, I, R, D, minVars) #set lin vars according to the min nonlin vars
    return linVars, minVars #return vars and linVars

def solveAllVars(S, A, I, R, D, nonLinConstraints, nonLinResol, printOut=False):
    linVars, nonLinVars = gridNonLinVars(S,A,I,R,D, nonLinConstraints, nonLinResol)

    if(printOut):
        print("Solution: ")
        print("b0: ", linVars[0])
        print("b1: ", linVars[1])
        print("k:  ", linVars[2])
        print("g:  ", linVars[3])
        print("nu: ", linVars[4])
        print("b2: ", nonLinVars[0])
        print("b3: ", nonLinVars[1])
        print("cost: ", getError(S,A,I,R,D, linVars, nonLinVars) )
        print() #spacer

    return linVars, nonLinVars


#--------------------------------------------------------------------------
    
    
    
    
#--------------------------------------------------------------------------  
def displayData(S,A,I,R,D, graphVals=[False,True,True,True,True]):
    
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(S, color="purple")
    if(graphVals[1]):
        ax.plot(A, color="orange")
    if(graphVals[2]):
        ax.plot(I, color="red")
    if(graphVals[3]):
        ax.plot(R, color="blue")
    if(graphVals[4]):
        ax.plot(D, color="black")
        
    return fig, ax
#--------------------------------------------------------------------------  





#------------------------------------------------------------------
#prediction functions
#predict the next some days using constant parameters, q and params will be calculated if not set, uses smoothing method  from paper
def calculateFuture(S,A,I,R,D, daysToPredict, params=None, nonLinParams=None):
    if(nonLinParams==None): #nonlinParams not calculate them
        varRanges = [(0,5000), (0,10)]
        varResol = [100, 10]
        params, nonLinParams = solveAllVars(S,A,I,R,D, varRanges, varResol)
    
    if(params==None): #nonLinParams are set but params aren't
        params=getLinVars(S,A,I,R,D, nonLinVars)
        
    print("Non Lin Vars:", nonLinParams)
    print("Lin Vars:", params)
    
    #set up matrices and starting info
    dt, X = getMatrix(S,A,I,R,D, nonLinParams)

    xPredict = np.zeros((len(X) + daysToPredict, np.shape(X)[1], np.shape(X)[2]))
    dtPredict = np.zeros((len(dt) + daysToPredict, np.shape(dt)[1], 1))

    xPredict[0:len(X)] = X
    dtPredict[0:len(dt)] = dt

    SP = np.zeros(len(S) + daysToPredict)
    AP = np.zeros(len(A) + daysToPredict)
    IP = np.zeros(len(I) + daysToPredict)
    RP = np.zeros(len(R) + daysToPredict)
    DP = np.zeros(len(D) + daysToPredict)

    SP[0:len(S)] = S 
    AP[0:len(A)] = A
    IP[0:len(I)] = I
    RP[0:len(R)] = R
    DP[0:len(D)] = D

    
    T = len(I) - 1
    for t in range(T-1, T + daysToPredict): #go from last element in known list to end of prediction, see paper for method
        pop = SP[t] + AP[t] + IP[t] + RP[t] + DP[t] #for normalizing b2*I
        
        newI = (IP+RP+DP)[t+1] - (IP+RP+DP)[t] 
        
        #populate the 5x5 matrix with parameters
        #susceptible row, dS = -(SI/S+I)
        xPredict[t,0,0] = -(SP[t] * AP[t]) / (SP[t] + AP[t]) #b0
        xPredict[t,0,1] = -(SP[t] * AP[t]) / (SP[t] + AP[t]) * (1 / (1 + (nonLinParams[0]*newI/pop)**nonLinParams[1] )) #b1

        #asympt row, dA = B*(S*I / S + I)
        xPredict[t,1,0] = (SP[t] * AP[t]) / (SP[t] + AP[t]) #b0
        xPredict[t,1,1] = (SP[t] * AP[t]) / (SP[t] + AP[t]) * (1 / (1 + (nonLinParams[0]*newI/pop)**nonLinParams[1] )) #b1
        xPredict[t,1,2] = -AP[t] #kappa
        
        xPredict[t,2,2] = AP[t] #kappa
        xPredict[t,2,3] = -IP[t] #gamma
        xPredict[t,2,4] = -IP[t] #nu

        #recovered row
        xPredict[t,3,3] = IP[t] #gamma

        #dead row
        xPredict[t,4,4] = IP[t] #nu

        #predict next iter matrix
        dtPredict[t,:,0] = (xPredict[t] @ params) 
        
        #find next SIRD, based on dtPredict[t] (which is S(t+1) - S(t)) to predict S(t) (and so on)
        SP[t+1] = SP[t] + dtPredict[t,0,0]
        AP[t+1] = AP[t] + dtPredict[t,1,0]
        IP[t+1] = IP[t] + dtPredict[t,2,0]
        RP[t+1] = RP[t] + dtPredict[t,3,0]
        DP[t+1] = DP[t] + dtPredict[t,4,0]
    
    return SP, AP, IP, RP, DP



#predict future days that are not known
def predictFuture(S,A,I,R,D, daysToPredict, linVars=None, nonLinVars=None, graphVals=[False,True,True,True]):
    pS, pA, pI, pR, pD = calculateFuture(S,A,I,R,D, daysToPredict, params=linVars, nonLinParams=nonLinVars)

    #plot actual and predicted values
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(S, color='purple', label='suscpetible')
        ax.plot(pS, color='purple', label='suscpetible', linestyle='dashed')
    if(graphVals[1]):
        ax.plot(A, color='orange', label='asyptomatic')
        ax.plot(pA, color='orange', label='asyptomatic', linestyle='dashed')
    if(graphVals[2]):
        ax.plot(I, color='red', label='infected')
        ax.plot(pI, color='red', label='infected', linestyle='dashed')
    if(graphVals[3]):
        ax.plot(R, color='blue', label='recovered')
        ax.plot(pR, color='blue', label='recovered', linestyle='dashed')
    if(graphVals[4]):
        ax.plot(D, color='black', label='dead')
        ax.plot(pD, color='black', label='dead', linestyle='dashed')
      
    return pS, pA, pI, pR, pD, fig, ax #for easy manipulation/graphing

    
#predict days that are known for testing purposes, predicts the end portion of the given data
def predictMatch(S,A,I,R,D, daysToPredict, linVars=None, nonLinVars=None, graphVals=[False,True,True,True]):
    pS, pA, pI, pR, pD = calculateFuture(S[0:-daysToPredict], A[0:-daysToPredict], I[0:-daysToPredict], R[0:-daysToPredict], D[0:-daysToPredict], daysToPredict, params=linVars, nonLinParams=nonLinVars)
    
    #plot actual and predicted values
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(S, color='purple', label='suscpetible')
        ax.plot(pS, color='purple', label='suscpetible', linestyle='dashed')
    if(graphVals[1]):
        ax.plot(A, color='orange', label='asyptomatic')
        ax.plot(pA, color='orange', label='asyptomatic', linestyle='dashed')
    if(graphVals[2]):
        ax.plot(I, color='red', label='infected')
        ax.plot(pI, color='red', label='infected', linestyle='dashed')
    if(graphVals[3]):
        ax.plot(R, color='blue', label='recovered')
        ax.plot(pR, color='blue', label='recovered', linestyle='dashed')
    if(graphVals[4]):
        ax.plot(D, color='black', label='dead')
        ax.plot(pD, color='black', label='dead', linestyle='dashed')
      
    return pS, pA, pI, pR, pD, fig, ax #for easy manipulation/graphing

#---------------------------------------------------------





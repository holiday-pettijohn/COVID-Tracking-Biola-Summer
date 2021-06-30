import numpy as np
import matplotlib.pyplot as plt

#this model is for the time varying parameter version of SIRD
# S' = -beta * (SI/S+I)
# A' = beta * (SI/S+I) - kappa*A
# I' = kappa*A - gamma*I - nu*I
# R' = gamma*I
# D' = nu*I

#--------------------------------------------------------------------------
#global variables used for many functions (no weight decay for time varying)
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
#create the basic model matrices
def getMatrix(S, A, I, R, D):    
    sirdMatrix = np.zeros((len(S) - 1, 5, 4))
    nextIterMatrix = np.zeros((len(S) - 1, 5, 1)) #the S(t+1), I(t+1), ... matrix
    
    #susceptible row, dS = 0
    sirdMatrix[:,0,0] = -(S[:-1] * A[:-1]) / (S[:-1] + A[:-1]) #beta

    #asymptomatic row
    sirdMatrix[:,1,0] = (S[:-1] * A[:-1]) / (S[:-1] + A[:-1]) #beta
    sirdMatrix[:,1,1] = -A[:-1] #kappa
    
    #infected row
    sirdMatrix[:,2,1] = A[:-1] #kappa
    sirdMatrix[:,2,2] = -I[:-1] #gamma
    sirdMatrix[:,2,3] = -I[:-1] #nu

    #recovered row
    sirdMatrix[:,3,2] = I[:-1] #gamma

    #dead row
    sirdMatrix[:,4,3] = I[:-1] #nu

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
def getLinVars(S, A, I, R, D, graph=False):
    nu = getNu(I,D)
    gamma = getGamma(I,R)
    kappa = getKappa(A,I,gamma,nu)
    beta = getBeta(S,A,I,kappa)
    
    if(graph):
        fig, ax = plt.subplots(4, 1, figsize=(18,8))
        ax[0].plot(beta, color="red")
        ax[1].plot(kappa, color="orange")
        ax[2].plot(gamma, color="blue")
        ax[3].plot(nu, color="black")
        ax[0].set_ylim(0)
        ax[1].set_ylim(0)
        ax[2].set_ylim(0)
        ax[3].set_ylim(0)
        
        return np.asarray([kappa, beta, gamma, nu]).transpose(), fig, ax
        
    return np.asarray([kappa, beta, gamma, nu]).transpose() #transpose so time is the first dimension


def getGamma(I, R):
    
    # R(t+1) - R(t) = gamma*I(t)
    y = R[1:] - R[:-1]
    x = I[:-1]

    return y/x #solve for gamma

def getNu(I, D):
    # D(t+1) - D(t) = nu*I(t)
    y = D[1:] - D[:-1]
    x = I[:-1]

    return y/x #solve for nu

def getKappa(A, I, gamma, nu):
    #A' = A*kapp - I*gmma - I*nu
    
    y = A[1:] - A[:-1] + gamma*I[:-1] + nu*I[:-1]
    x = A[:-1]
    
    return y/x #solve for kappa
    
def getBeta(S, A, I, kappa):
    y = np.zeros((len(I)-1,2,1))
    x = np.zeros((len(I)-1,2,1)) 

    beta = np.zeros(len(I) - 1)
    
    #betaNonLin = [b2,b3]
    #dS = -beta * (SI/S+I)
    #dI = beta * (SI/S+I) - kappa*A
    y[:,0,0] = (S[1:] - S[:-1]) #s row
    y[:,1,0] = (I[1:] - I[:-1]) + kappa*A[:-1]
    
    x[:,0,0] = -(S[:-1]*A[:-1]) / (S[:-1] + A[:-1])
    x[:,1,0] = (S[:-1]*A[:-1]) / (S[:-1] + A[:-1])
    
    for t in range(len(y)):
        beta[t] = np.linalg.lstsq(x[t], y[t], rcond = None)[0].flatten()[0]
        
    return beta

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
def calculateFuture(S,A,I,R,D, daysToPredict, params=None):
    if(params==None):
        params=getLinVars(S,A,I,R,D)
    
    #average out final days (weighted average) instead of just using the last day
    paramsTemp = params
    params = paramsTemp[0]
    for i in range(1,len(paramsTemp)):
        params = params*.5 + paramsTemp[i]*.5 #weighted so last day is 50%, 2nd to last is 25% and so 
    print("Lin Vars:", params)
    
    #set up matrices and starting info
    dt, X = getMatrix(S,A,I,R,D)

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
    for t in range(T, T + daysToPredict): #go from last element in known list to end of prediction, see paper for method
        #populate the 5x5 matrix with parameters
        #susceptible row, dS = -(SI/S+I)
        xPredict[t,0,0] = -(SP[t] * AP[t]) / (SP[t] + AP[t])

        #asymptomatic row
        xPredict[t,1,0] = (SP[t] * AP[t]) / (SP[t] + AP[t])
        xPredict[t,1,1] = -AP[t] #kappa
        
        #infected row, dI = kappa*A - I*gamma - nu*I
        xPredict[t,2,1] = AP[t] #kappa
        xPredict[t,2,2] = -IP[t] #gamma
        xPredict[t,2,3] = -IP[t] #nu

        #recovered row
        xPredict[t,3,2] = IP[t] #gamma

        #dead row
        xPredict[t,4,3] = IP[t] #nu

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
def predictFuture(S,A,I,R,D, daysToPredict, linVars=None, graphVals=[False,True,True,True,True]):
    pS, pA, pI, pR, pD = calculateFuture(S,A,I,R,D, daysToPredict, params=linVars)

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
def predictMatch(S,A,I,R,D, daysToPredict, linVars=None, graphVals=[False,True,True,True]):
    pS, pA, pI, pR, pD = calculateFuture(S[0:-daysToPredict], A[0:-daysToPredict], I[0:-daysToPredict], R[0:-daysToPredict], D[0:-daysToPredict], daysToPredict, params=linVars)
    
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





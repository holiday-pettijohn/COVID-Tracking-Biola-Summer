import numpy as np
import matplotlib.pyplot as plt

#this model is for the constant parameter version of SIRD
# S' = -beta * (SA/S+A)
# A' = beta * (SA/S+A) - kappa*A
# I' = kappa*A - gamma*I - nu*I
# R' = gamma*I
# D' = nu*I

#--------------------------------------------------------------------------
#global variables used for many functions
regularizer = 0
weightDecay = 1
#--------------------------------------------------------------------------

#to calculate A
# A_total(t) = I_total(t + shift) #anyone who is infected was asymptomatic shifted days ago
# A_total = A + I + R + D
# I_total = I + R + D
# therefore: A = A_total - (I+R+D)
# A(t) = I_total(t+shift) - I_total(t)

def getAsympt(I,R,D, shift=10): #assume the current infected population was the asymptomatic population shifted days ago
    #totalI = I + R + D
    
    #A = totalI[shift:] - totalI[:-shift] #I_total(t) - I_total(t+shift)
    A = I[shift:]
    #no definition of A on range t-shift to end, so returns a smaller size than the given I, R, D
    
    return A

def getSuscept(A, I, R, D, q, pop):
    return (q*pop) - A - I - R - D #S + I + R + D = q*pop

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
def getLinVars(S, A, I, R, D):
    nu = getNu(I,D)
    gamma = getGamma(I,R)
    kappa = getKappa(A,I,gamma,nu)
    beta = getBeta(S,A,I,kappa)
    
    return [beta, kappa, gamma, nu]


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

def getKappa(A,I, gamma, nu):
    #A' = A*kapp - I*gmma - I*nu
    y = np.zeros((len(I)-1,1))
    x = np.zeros((len(I)-1,1))
    
    y[:,0] = A[1:] - A[:-1] + gamma*I[:-1] + nu*I[:-1]
    x[:,0] = A[:-1]
    
    #add weight decay
    T = len(I)-1
    for t in range(T):
        y[t] = y[t] * np.sqrt(weightDecay**(T-t))
        x[t] = x[t] * np.sqrt(weightDecay**(T-t))
    
    return np.linalg.lstsq(x, y, rcond = None)[0].flatten()[0] #solve for kappa
    
def getBeta(S, A, I, kappa):
    y = np.zeros((2*(len(I)-1),1)) # 2 times length since every other row is for S' and every other is I'
    x = np.zeros((2*(len(I)-1),1))
    
    #betaNonLin = [b2,b3]
    #dS = -beta * (SI/S+I)
    #dI = beta * (SI/S+I) - kappa*A
    
    y[::2,  0] = (S[1:] - S[:-1]) #::2 is for skipping every other row (starts at 0)
    y[1::2, 0] = (I[1:] - I[:-1]) + kappa*A[:-1]
   
    x[::2,  0] = -(S[:-1]*A[:-1]) / (S[:-1] + A[:-1])
    x[1::2, 0] = (S[:-1]*A[:-1]) / (S[:-1] + A[:-1]) 
    
    #add weight decay
    T = len(I)-1
    for t in range(T):
        x[t*2:(t*2)+2] = x[t*2:(t*2)+2] * np.sqrt(weightDecay**(T - t))
        y[t*2:(t*2)+2] = y[t*2:(t*2)+2] * np.sqrt(weightDecay**(T - t))
    
    return np.linalg.lstsq(x, y, rcond = None)[0].flatten()[0] #solve for beta
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------
#find the error of the current parameters
def getError(S, A, I, R, D, linVars, regError=True): #the custom error function for SIRD    
    y, x = getMatrix(S, A, I, R, D)
    
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


#--------------------------------------------------------------------------
#solve for q by gridding
def getQ(A, I, R, D, pop, qMax = 1, resol=100, graph=False):
    qMin = max((I + R + D)/pop)
    
    qList = np.zeros(resol)
    for i in range(resol):
        qList[i] = qMin + (i/resol)*(qMax-qMin) #go from qMin to qMax
        
    errorList = [] #find the error for each calculated value of q
    for q in qList: #check eeach value of q
        S = q*pop - A - I - R - D
        #normalize so that S+A+I+R+D = 1, this allows errors to be consistant between different q values
        S = S/(q*pop)
        A = A/(q*pop)
        I = I/(q*pop)
        R = R/(q*pop)
        D = D/(q*pop)      
        
        linVars = getLinVars(S,A,I,R,D)
        errorList.append(getError(S,A,I,R,D, linVars, regError=False)) #add errror, don't do a regularization error
    
    bestQIndex = 0
    for i in range(resol):
        if(errorList[bestQIndex] > errorList[i]):
            bestQIndex = i
    if(graph):
        #plot objective function with q on the x-axis
        fig, ax = plt.subplots(figsize=(18,8))
        ax.plot(qList, errorList, color='purple')  
        return qList[bestQIndex], fig,ax #return figure for modification

    return qList[bestQIndex]
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
        
    for t in range(T): #individual day error plotting
        #populate the 5x5 matrix with parameters
        #susceptible row, dS = -(SI/S+I)
        xPredict[t,0,0] = -(S[t] * A[t]) / (S[t] + A[t])

        #asymptomatic row
        xPredict[t,1,0] = (S[t] * A[t]) / (S[t] + A[t])
        xPredict[t,1,1] = -A[t] #kappa
        
        #infected row, dI = kappa*A - I*gamma - nu*I
        xPredict[t,2,1] = A[t] #kappa
        xPredict[t,2,2] = -I[t] #gamma
        xPredict[t,2,3] = -I[t] #nu

        #recovered row
        xPredict[t,3,2] = I[t] #gamma

        #dead row
        xPredict[t,4,3] = I[t] #nu

        #predict next iter matrix
        dtPredict[t,:,0] = (xPredict[t] @ params) 
        
        #find next SIRD, based on dtPredict[t] (which is S(t+1) - S(t)) to predict S(t) (and so on)
        SP[t+1] = S[t] + dtPredict[t,0,0]
        AP[t+1] = A[t] + dtPredict[t,1,0]
        IP[t+1] = I[t] + dtPredict[t,2,0]
        RP[t+1] = R[t] + dtPredict[t,3,0]
        DP[t+1] = D[t] + dtPredict[t,4,0]
    
    
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
      
    ax.axvline(len(S), color='black', linestyle='dashed')
    
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
      
    ax.axvline(len(S)-daysToPredict, color='black', linestyle='dashed')
    
    return pS, pA, pI, pR, pD, fig, ax #for easy manipulation/graphing

#---------------------------------------------------------





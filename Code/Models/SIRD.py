import numpy as np
import matplotlib.pyplot as plt

#this model is for the constant parameter version of SIRD
# S' = -beta * (SI/S+I)
# I' = beta * (SI/S+I) - gamma*I - nu*I
# R' = gamma*I
# D' = nu*I

#--------------------------------------------------------------------------
#global variables used for many functions
regularizer = 0
weightDecay = 1
#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
#create the basic model matrices
def getMatrix(S, I, R, D):    
    sirdMatrix = np.zeros((len(S) - 1, 4, 3))
    nextIterMatrix = np.zeros((len(S) - 1, 4, 1)) #the S(t+1), I(t+1), ... matrix
    
    #susceptible row, dS = 0
    sirdMatrix[:,0,0] = -(S[:-1] * I[:-1]) / (S[:-1] + I[:-1]) #beta

    #infected row
    sirdMatrix[:,1,0] = (S[:-1] * I[:-1]) / (S[:-1] + I[:-1]) #beta
    sirdMatrix[:,1,1] = -I[:-1] #gamma
    sirdMatrix[:,1,2] = -I[:-1] #nu

    #recovered row
    sirdMatrix[:,2,1] = I[:-1] #gamma

    #dead row
    sirdMatrix[:,3,2] = I[:-1] #nu

    #populate the S(t+1), I(t+1), ... matrix
    nextIterMatrix[:,0,0] = S[1:] - S[:-1]
    nextIterMatrix[:,1,0] = I[1:] - I[:-1]
    nextIterMatrix[:,2,0] = R[1:] - R[:-1]
    nextIterMatrix[:,3,0] = D[1:] - D[:-1]

    return nextIterMatrix, sirdMatrix
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
#solve for parameters using weight decay and solving row by row
def getLinVars(S, I, R, D):
    nu = getNu(I,D)
    gamma = getGamma(I,R)
    beta = getBeta(S,I,gamma,nu)
    
    return [beta, gamma, nu]


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

def getBeta(S, I, gamma, nu):
    y = np.zeros((2*(len(I)-1),1)) # 2 times length since every other row is for S' and every other is I'
    x = np.zeros((2*(len(I)-1),1))
    
    #betaNonLin = [b2,b3]
    #dS = -beta * (SI/S+I)
    #dI = beta * (SI/S+I)\
    
    y[::2,  0] = (S[1:] - S[:-1]) #::2 is for skipping every other row (starts at 0)
    y[1::2, 0] = (I[1:] - I[:-1]) + gamma*I[:-1] + nu*I[:-1]
    
    x[::2,  0] = -(S[:-1]*I[:-1]) / (S[:-1] + I[:-1])
    x[1::2, 0] = (S[:-1]*I[:-1]) / (S[:-1] + I[:-1]) 
    
    #add weight decay
    T = len(I)-1
    for t in range(T):
        x[t*2:(t*2)+2] = x[t*2:(t*2)+2] * np.sqrt(weightDecay**(T - t))
        y[t*2:(t*2)+2] = y[t*2:(t*2)+2] * np.sqrt(weightDecay**(T - t))
    
    return np.linalg.lstsq(x, y, rcond = None)[0].flatten()[0] #solve for beta
#--------------------------------------------------------------------------



#--------------------------------------------------------------------------
#find the error of the current parameters
def getError(S, I, R, D, linVars, regError=True): #the custom error function for SIRD    
    y, x = getMatrix(S, I, R, D)
    
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
def getQ(I, R, D, pop, qMax = 1, resol=100, graph=False):
    qMin = max((I + R + D)/pop)
    
    qList = np.zeros(resol)
    for i in range(resol):
        qList[i] = qMin + (i/resol)*(qMax-qMin) #go from qMin to qMax
        
    errorList = [] #find the error for each calculated value of q
    for q in qList: #check eeach value of q
        S = q*pop - I - R - D
        #normalize so that S+I+R+D = 1, this allows errors to be consistant between different q values
        S = S/(q*pop)
        I = I/(q*pop)
        R = R/(q*pop)
        D = D/(q*pop)      
        
        linVars = getLinVars(S,I,R,D)
        errorList.append(getError(S,I,R,D, linVars, regError=False)) #add errror, don't do a regularization error
    
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
def displayData(S,I,R,D, graphVals=[False,True,True,True]):
    
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(S, color="purple")
    if(graphVals[1]):
        ax.plot(I, color="red")
    if(graphVals[2]):
        ax.plot(R, color="blue")
    if(graphVals[3]):
        ax.plot(D, color="black")
        
    return fig, ax
#--------------------------------------------------------------------------  







#------------------------------------------------------------------
#prediction functions
#predict the next some days using constant parameters, q and params will be calculated if not set, uses smoothing method  from paper
def calculateFuture(S,I,R,D, daysToPredict, params=None):
    if(params==None):
        params=getLinVars(S,I,R,D)
    print("Lin Vars:", params)
    
    #set up matrices and starting info
    dt, X = getMatrix(S,I,R,D)

    xPredict = np.zeros((len(X) + daysToPredict, np.shape(X)[1], np.shape(X)[2]))
    dtPredict = np.zeros((len(dt) + daysToPredict, np.shape(dt)[1], 1))

    xPredict[0:len(X)] = X
    dtPredict[0:len(dt)] = dt

    SP = np.zeros(len(S) + daysToPredict)
    IP = np.zeros(len(I) + daysToPredict)
    RP = np.zeros(len(R) + daysToPredict)
    DP = np.zeros(len(D) + daysToPredict)

    SP[0:len(S)] = S 
    IP[0:len(I)] = I
    RP[0:len(R)] = R
    DP[0:len(D)] = D

    T = len(I) - 1
    
    for t in range(T, T + daysToPredict): #go from last element in known list to end of prediction, see paper for method
        #populate the 5x5 matrix with parameters
        #susceptible row, dS = -(SI/S+I)
        xPredict[t,0,0] = -(SP[t] * IP[t]) / (SP[t] + IP[t])

        #infected row, dA = B*(S*I / S + I)
        xPredict[t,1,0] = (SP[t] * IP[t]) / (SP[t] + IP[t]) #b0
        xPredict[t,1,1] = -IP[t] #gamma
        xPredict[t,1,2] = -IP[t] #nu

        #recovered row
        xPredict[t,2,1] = IP[t] #gamma

        #dead row
        xPredict[t,3,2] = IP[t] #nu

        #predict next iter matrix
        dtPredict[t,:,0] = (xPredict[t] @ params) 
        
        #find next SIRD, based on dtPredict[t] (which is S(t+1) - S(t)) to predict S(t) (and so on)
        SP[t+1] = SP[t] + dtPredict[t,0,0]
        IP[t+1] = IP[t] + dtPredict[t,1,0]
        RP[t+1] = RP[t] + dtPredict[t,2,0]
        DP[t+1] = DP[t] + dtPredict[t,3,0]
    
    #find corrective plotting error
    for t in range(T): #go from last element in known list to end of prediction, see paper for method
        #populate the 5x5 matrix with parameters
        #susceptible row, dS = -(SI/S+I)
        xPredict[t,0,0] = -(S[t] * I[t]) / (S[t] + I[t])

        #infected row, dA = B*(S*I / S + I)
        xPredict[t,1,0] = (S[t] * I[t]) / (S[t] + I[t]) #b0
        xPredict[t,1,1] = -I[t] #gamma
        xPredict[t,1,2] = -I[t] #nu

        #recovered row
        xPredict[t,2,1] = I[t] #gamma

        #dead row
        xPredict[t,3,2] = I[t] #nu

        #predict next iter matrix
        dtPredict[t,:,0] = (xPredict[t] @ params) 
        
        #find next SIRD, based on dtPredict[t] (which is S(t+1) - S(t)) to predict S(t) (and so on)
        SP[t+1] = S[t] + dtPredict[t,0,0]
        IP[t+1] = I[t] + dtPredict[t,1,0]
        RP[t+1] = R[t] + dtPredict[t,2,0]
        DP[t+1] = D[t] + dtPredict[t,3,0]
    
    return SP, IP, RP, DP



#predict future days that are not known
def predictFuture(S,I,R,D, daysToPredict, linVars=None, graphVals=[False,True,True,True], graph=True):
    pS, pI, pR, pD = calculateFuture(S,I,R,D, daysToPredict, params=linVars)

    if(graph==False):
        return pS, pI, pR, pD
    
    #plot actual and predicted values
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(S, color='purple', label='suscpetible')
        ax.plot(pS, color='purple', label='suscpetible', linestyle='dashed')
    if(graphVals[1]):
        ax.plot(I, color='red', label='infected')
        ax.plot(pI, color='red', label='infected', linestyle='dashed')
    if(graphVals[2]):
        ax.plot(R, color='blue', label='recovered')
        ax.plot(pR, color='blue', label='recovered', linestyle='dashed')
    if(graphVals[3]):
        ax.plot(D, color='black', label='dead')
        ax.plot(pD, color='black', label='dead', linestyle='dashed')
      
    ax.axvline(len(S), color='black', linestyle='dashed')
    
    return pS, pI, pR, pD, fig, ax #for easy manipulation/graphing

    
#predict days that are known for testing purposes, predicts the end portion of the given data
def predictMatch(S,I,R,D, daysToPredict, linVars=None, graphVals=[False,True,True,True], graph=True):
    pS, pI, pR, pD = calculateFuture(S[0:-daysToPredict], I[0:-daysToPredict], R[0:-daysToPredict], D[0:-daysToPredict], daysToPredict, params=linVars)
    
    if(graph==False):
        return pS, pI, pR, pD
    
    #plot actual and predicted values
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(S, color='purple', label='suscpetible')
        ax.plot(pS, color='purple', label='suscpetible', linestyle='dashed')
    if(graphVals[1]):
        ax.plot(I, color='red', label='infected')
        ax.plot(pI, color='red', label='infected', linestyle='dashed')
    if(graphVals[2]):
        ax.plot(R, color='blue', label='recovered')
        ax.plot(pR, color='blue', label='recovered', linestyle='dashed')
    if(graphVals[3]):
        ax.plot(D, color='black', label='dead')
        ax.plot(pD, color='black', label='dead', linestyle='dashed')
      
    ax.axvline(len(S)-daysToPredict, color='black', linestyle='dashed')
    
    return pS, pI, pR, pD, fig, ax #for easy manipulation

#---------------------------------------------------------





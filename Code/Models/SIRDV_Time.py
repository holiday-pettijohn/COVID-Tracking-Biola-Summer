import numpy as np
import matplotlib.pyplot as plt

#this model is for the time varying parameter version of SIRD
# S' = -beta * (SI/S+I)
# I' = beta * (SI/S+I) - gamma*I - nu*I
# R' = gamma*I
# D' = nu*I

#--------------------------------------------------------------------------
#global variables used for many functions (no weight decay since this is time varying)

#--------------------------------------------------------------------------

#--------------------------------------------------------------------------
#create the basic model matrices
def getMatrix(S, I, R, D, V):    
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
    nextIterMatrix[:,0,0] = S[1:] - S[:-1] + ( (V[1:] - V[:-1]) * S[:-1] / (S[:-1] + R[:-1]) ) #V'S/S+R
    nextIterMatrix[:,1,0] = I[1:] - I[:-1]
    nextIterMatrix[:,2,0] = R[1:] - R[:-1] + ( (V[1:] - V[:-1]) * R[:-1] / (S[:-1] + R[:-1]) ) #V'S/S+R
    nextIterMatrix[:,3,0] = D[1:] - D[:-1]

    return nextIterMatrix, sirdMatrix
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
#solve for parameters using weight decay and solving row by row
def getLinVars(S, I, R, D, V, graph=False):
    nu = getNu(I,D)
    gamma = getGamma(S,I,R,V)
    beta = getBeta(S,I,R,V,gamma,nu)
    
    if(graph):
        fig, ax = plt.subplots(3, 1, figsize=(18,8))
        ax[0].plot(beta, color="red")
        ax[1].plot(gamma, color="blue")
        ax[2].plot(nu, color="black")
        ax[0].set_ylim(0)
        ax[1].set_ylim(0)
        ax[2].set_ylim(0)
        return np.asarray([beta, gamma, nu]).transpose(), fig, ax
        
    return np.asarray([beta, gamma, nu]).transpose() #transpose so time is the first dimension


def getGamma(S, I, R,V):
    # R(t+1) - R(t) = gamma*I(t)
    y = R[1:] - R[:-1] + ( (V[1:] - V[:-1]) * R[:-1] / (S[:-1] + R[:-1]) ) #V'S/S+R
    x = I[:-1]

    return y/x #solve for gamma

def getNu(I, D):
    # D(t+1) - D(t) = nu*I(t)
    y = D[1:] - D[:-1]
    x = I[:-1]

    return y/x #solve for nu

def getBeta(S, I, R,V,gamma, nu):
    y = np.zeros((len(I)-1,2,1))
    x = np.zeros((len(I)-1,2,1)) 
    
    beta = np.zeros(len(I) - 1)
    
    #betaNonLin = [b2,b3]
    #dS = -beta * (SI/S+I)
    #dI = beta * (SI/S+I)
    y[:,0,0] = (S[1:] - S[:-1]) + ( (V[1:] - V[:-1]) * S[:-1] / (S[:-1] + R[:-1]) ) #V'S/S+R #s row
    y[:,1,0] = (I[1:] - I[:-1]) + gamma*I[:-1] + nu*I[:-1] #i row
    
    x[:,0,0] = -(S[:-1]*I[:-1]) / (S[:-1] + I[:-1])
    x[:,1,0] = (S[:-1]*I[:-1]) / (S[:-1] + I[:-1])
    
    for t in range(len(y)):
        beta[t] = np.linalg.lstsq(x[t], y[t], rcond = None)[0].flatten()[0]
    
    return beta #solve for beta
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------  
def displayData(S,I,R,D, V, graphVals=[False,True,True,True,True]):
    
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphVals[0]):
        ax.plot(S, color="purple")
    if(graphVals[1]):
        ax.plot(I, color="red")
    if(graphVals[2]):
        ax.plot(R, color="blue")
    if(graphVals[3]):
        ax.plot(D, color="black")
    if(graphVals[4]):
        ax.plot(V, color="green")
    
    return fig, ax
#--------------------------------------------------------------------------  







#------------------------------------------------------------------
#prediction functions
#predict the next some days using constant parameters, q and params will be calculated if not set, uses smoothing method  from paper
def calculateFuture(S,I,R,D, V, daysToPredict, params=None):
    if(params==None):
        params=getLinVars(S,I,R,D, V)
        
    #average out final days (weighted average) instead of just using the last day
    paramsTemp = params
    params = paramsTemp[0]
    for i in range(1,len(paramsTemp)):
        params = params*.5 + paramsTemp[i]*.5 #weighted so last day is 50%, 2nd to last is 25% and so 
    print("Lin Vars:", params)
    
    #set up matrices and starting info
    dt, X = getMatrix(S,I,R,D, V)

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
        SP[t+1] = SP[t] + dtPredict[t,0,0] - ( (V[t+1] - V[t]) * SP[t] / (SP[t] + RP[t]) ) #V'S/S+R
        IP[t+1] = IP[t] + dtPredict[t,1,0]
        RP[t+1] = RP[t] + dtPredict[t,2,0] - ( (V[t+1] - V[t]) * RP[t] / (SP[t] + RP[t]) ) #V'S/S+R
        DP[t+1] = DP[t] + dtPredict[t,3,0]
    
    return SP, IP, RP, DP



#predict future days that are not known
def predictFuture(S,I,R,D, V, daysToPredict, linVars=None, graphVals=[False,True,True,True]):
    pS, pI, pR, pD = calculateFuture(S,I,R,D, V, daysToPredict, params=linVars)

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
    if(graphVals[4]):
        ax.plot(V, color='green', label="vaccinations")
      
    return pS, pI, pR, pD, fig, ax #for easy manipulation/graphing

    
#predict days that are known for testing purposes, predicts the end portion of the given data
def predictMatch(S,I,R,D, V, daysToPredict, linVars=None, graphVals=[False,True,True,True]):
    pS, pI, pR, pD = calculateFuture(S[0:-daysToPredict], I[0:-daysToPredict], R[0:-daysToPredict], D[0:-daysToPredict], V, daysToPredict, params=linVars)
    
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
    if(graphVals[4]):
        ax.plot(V, color='green', label="vaccinations")
      
    return pS, pI, pR, pD, fig, ax #for easy manipulation

#---------------------------------------------------------





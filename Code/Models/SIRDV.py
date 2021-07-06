import numpy as np
import matplotlib.pyplot as plt


import csv

#this model is for the constant parameter version of SIRD
# S' = -beta * (SI/S+I) - V'*S/S+R
# I' = beta * (SI/S+I) - gamma*I - nu*I
# R' = gamma*I - V'*R/S+R
# D' = nu*I

#--------------------------------------------------------------------------
#global variables used for many functions
regularizer = 0
weightDecay = 1
#--------------------------------------------------------------------------


def approxSusceptRecov(S,R,V):
    
    for t in range(1, len(S)): 

        subS = ((V[t] - V[t-1]) * S[t]) / (S[t] + R[t])
        subR = ((V[t] - V[t-1]) * R[t]) / (S[t] + R[t])

        for i in range(t,len(S)):
            S[i] -= subS
            R[i] -= subR
    return S,R

#normal load function but also return vaccinations
def loadData(filename):
    csvfile=open(filename, newline='', encoding='UTF-8')
    rd = csv.reader(csvfile, delimiter=',')
    data=[]
    for lv in rd: #generating the data matrix
        data.append(lv)
    header = data[0] #get the labels
    infectionData=(data[1:]) #data without the labels
    infectionData = np.array(infectionData)
    dates = infectionData[:,header.index("Dates")]
    infected = infectionData[:,header.index("Infected")]
    recovered = infectionData[:,header.index("Recovered")]
    deaths = infectionData[:,header.index("Deaths")]
    vacc = infectionData[:,header.index("Vaccinated")]
    vacc = vacc.astype(float)
    deaths = deaths.astype(float)
    recovered = recovered.astype(float)
    infected = infected.astype(float)
    return dates, infected, recovered, deaths, vacc



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
    nextIterMatrix[:,2,0] = R[1:] - R[:-1] + ( (V[1:] - V[:-1]) * R[:-1] / (S[:-1] + R[:-1]) ) #V'R/S+R
    nextIterMatrix[:,3,0] = D[1:] - D[:-1]

    return nextIterMatrix, sirdMatrix
#--------------------------------------------------------------------------


#--------------------------------------------------------------------------
#solve for parameters using weight decay and solving row by row
def getLinVars(S, I, R, D, V):
    nu = getNu(I,D)
    gamma = getGamma(S,I,R, V)
    beta = getBeta(S,I,R,V, gamma,nu)
    
    return [beta, gamma, nu]


def getGamma(S, I, R, V):
    y = np.zeros((len(I)-1,1))
    x = np.zeros((len(I)-1,1))
    
    # R(t+1) - R(t) = gamma*I(t)
    y[:,0] = R[1:] - R[:-1] + ( (V[1:] - V[:-1]) * R[:-1] / (S[:-1] + R[:-1]) ) #V'R/S+R
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

def getBeta(S, I, R, V, gamma, nu):
    y = np.zeros((2*(len(I)-1),1)) # 2 times length since every other row is for S' and every other is I'
    x = np.zeros((2*(len(I)-1),1))
    
    #betaNonLin = [b2,b3]
    #dS = -beta * (SI/S+I)
    #dI = beta * (SI/S+I)\
    
    y[::2,  0] = (S[1:] - S[:-1]) + ( (V[1:] - V[:-1]) * S[:-1] / (S[:-1] + R[:-1]) ) #::2 is for skipping every other row 
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
def getError(S, I, R, D, V, linVars, regError=True): #the custom error function for SIRD    
    y, x = getMatrix(S, I, R, D, V)
    
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
def getQ(I, R, D, V, pop, qMax = 1, resol=100, graph=False):
    qMin = max((I + R + D + V)/pop)
    
    qList = np.zeros(resol)
    for i in range(resol):
        qList[i] = qMin + (i/resol)*(qMax-qMin) #go from qMin to qMax
        
    errorList = [] #find the error for each calculated value of q
    for q in qList: #check eeach value of q
        S = q*pop - I - R - D - V
        #normalize so that S+I+R+D = 1, this allows errors to be consistant between different q values
        S = S/(q*pop)
        I = I/(q*pop)
        R = R/(q*pop)
        D = D/(q*pop)  
        V = V/(q*pop)
        
        linVars = getLinVars(S,I,R,D, V)
        errorList.append(getError(S,I,R,D, V, linVars, regError=False)) #add errror, don't do a regularization error
    
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
def displayData(S,I,R,D,V, graphVals=[False,True,True,True,True]):
    
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
#note that the V given should be equal to len(S) + daysToPredict + 1
def calculateFuture(S,I,R,D,V, daysToPredict, params=None):
    if(params==None):
        params=getLinVars(S,I,R,D,V[:len(S)])
    print("Lin Vars:", params)
    
    #set up matrices and starting info
    dt, X = getMatrix(S,I,R,D,V[:len(S)])

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
        SP[t+1] = SP[t] + dtPredict[t,0,0] - ( (V[t+1] - V[t]) * SP[t] / (SP[t] + RP[t]) )
        IP[t+1] = IP[t] + dtPredict[t,1,0]
        RP[t+1] = RP[t] + dtPredict[t,2,0] - ( (V[t+1] - V[t]) * RP[t] / (SP[t] + RP[t]) )
        DP[t+1] = DP[t] + dtPredict[t,3,0]
        
    for t in range(0, T): #go from last element in known list to end of prediction, see paper for method
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
        SP[t+1] = S[t] + dtPredict[t,0,0] - ( (V[t+1] - V[t]) * S[t] / (S[t] + R[t]) )
        IP[t+1] = I[t] + dtPredict[t,1,0]
        RP[t+1] = R[t] + dtPredict[t,2,0] - ( (V[t+1] - V[t]) * R[t] / (S[t] + R[t]) )
        DP[t+1] = D[t] + dtPredict[t,3,0]
    
    return SP, IP, RP, DP



#predict future days that are not known
def predictFuture(S,I,R,D,V, daysToPredict, linVars=None, graphVals=[False,True,True,True,True], graph=True):
    pS, pI, pR, pD = calculateFuture(S,I,R,D,V, daysToPredict, params=linVars)

    if(not graph):
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
    if(graphVals[4]):
        ax.plot(V, color='green', label="vaccinations")
      
    return pS, pI, pR, pD, fig, ax #for easy manipulation/graphing

    
#predict days that are known for testing purposes, predicts the end portion of the given data
def predictMatch(S,I,R,D,V, daysToPredict, linVars=None, graphVals=[False,True,True,True,True], graph=True):
    pS, pI, pR, pD = calculateFuture(S[0:-daysToPredict], I[0:-daysToPredict], R[0:-daysToPredict], D[0:-daysToPredict], V, daysToPredict, params=linVars)
    
    if(not graph):
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
    if(graphVals[4]):
        ax.plot(V, color='green', label="vaccinations")
      
    return pS, pI, pR, pD, fig, ax #for easy manipulation

#---------------------------------------------------------





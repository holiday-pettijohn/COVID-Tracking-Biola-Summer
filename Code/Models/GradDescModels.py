import numpy as np
import matplotlib.pyplot as plt
import random
import scipy.optimize as opt

##################################################### Functions for normal fitting
def graphParams(params, consts, graphA=False):
    A, I = simFunc(params, consts, giveA=True)
    
    fig, ax = plt.subplots(figsize=(18,8))
    if(graphA):
        ax.plot(A, color="orange", linestyle="dashed", alpha=.7)
    ax.plot(I, color="red")
    return fig, ax

def startFunc(consts):
    params = np.zeros(8)
    params[0] = .001 + consts[1] #A(0), between 0 and 10 percent of the population
    params[1] = consts[1] #10e-10 + random.random()*.01 #I(0), between 0 and .1 percent of the population, avoid 0
    params[2] = .1 + random.random()*.4 #gamma0, asympt->recovs + infected - floor asympt infect rate
    params[3] = .001 + random.random()*.1 #gamma1, asympt->infected
    params[4] = .001 + random.random()*.1 #nu, infected -> recov or dead
    params[5] = .3 + random.random()*1.5 #beta0, ceiling infectRate - floor infect rate
    params[6] = 13 + 125*random.random() #beta1, scaler on feedback
    params[7] = .5 + random.random()*2 #beta2, exponential [.5, 5.5]
    
    return params


def simFunc(params, consts, giveA=False): #option to return A and I

        dayNum = consts[0]

        A = np.zeros((dayNum))
        I = np.zeros((dayNum))

        A[0] = params[0]
        I[0] = params[1]

        gamma0 = params[2]
        gamma1 = params[3]
        nu = params[4]

        beta = params[5:8]

        #print(params)

        #iterate the arrays using the definition K' and I'
        for t in range(len(I)-1): #define I and K on range [1, length)
            diffA = (beta[0]/ (1 + (beta[1]*I[t])**(beta[2]) ) )*A[t] - gamma0*A[t]
            diffI = gamma1*A[t] - nu*I[t]

            A[t+1] = diffA + A[t]
            I[t+1] = diffI + I[t]

        if(giveA):
            return A,I

        return I #I is the data generated


#x is the starting params, args = (consts, y)
def errFunc(params, consts, normalWeight, slopeWeight, wDecay, lastDay, skip, y):
    x = simFunc(params, consts)
    
    error = 0
    if(normalWeight!=0):
        for t in range(skip,len(y)):
            error = error + ((y[t] - x[t])**2)*wDecay**(len(y)-t+1) #squared error
        error = error / len(y) # / T, average error
        
        error = error + lastDay * ((y[-1] - x[-1])**2) #add error of the last day
    
    
    slopeError = 0
    if(slopeWeight!=0):
        dy = np.diff(y)
        dx = np.diff(x)
        for t in range(skip,len(dy)):
            slopeError = slopeError +  ((dy[t] - dx[t])**2)*wDecay**(len(dy)-t+1) #squared error
        slopeError = slopeError / len(dy) # / T, average error
        
        slopeError = slopeError + lastDay * ((dy[-1] - dx[-1])**2) #add error of the last day

    return error*normalWeight + slopeError*slopeWeight



def getParams(I, consts, normalWeight=1, slopeWeight=0, wDecay=1, lastDay=0, skip=0, randomIterCount=100, method="SLSQP"):
    bestParams = startFunc(consts)
    bestError = 10e10 #arbitrary large value
    
    for i in range(randomIterCount):
        print("Iter: ", i, end="")
        newParams = startFunc(consts)
        newParams = opt.minimize(errFunc, newParams, (consts, normalWeight, slopeWeight, wDecay, lastDay, skip, I), method=method)['x']
        newError = errFunc(newParams, consts, normalWeight, slopeWeight, wDecay, lastDay, skip, I)
    
        print("\r               \r", end="") #go back to the start of the line and write over
        if(newError < bestError):
            bestError = newError
            bestParams = newParams
            print(i, "New best error: ", bestError)
            
    return bestParams


def getStablePoint(params, consts):
    #I stable point = (b0/gamm0 - 1)^(1/b2) / b1
    #A stable point = nu/gamma1 * (b0/gamm0 - 1)^(1/b2) / b1
    
    gamma0 = params[2]
    gamma1 = params[3]
    nu = params[4]
    beta = params[5:]
    
    return (beta[0]/gamma0 - 1)**(1/beta[2]) / beta[1]

##################################################### END ### Functions for normal fitting


##################################################### Functions for fitting with multiple b1s

def refineParamsB1(I, simpleParams, consts, normalWeight=1, slopeWeight=0, wDecay=1, skip=0, method="SLSQP"):
    newParams = np.zeros(len(simpleParams)+len(consts[3]))
    
    newParams[0:6] = simpleParams[0:6] #copy from A(0) to b0
    
    for i in range(len(consts[3])): #copy b1 over to the multiple b1's
        newParams[6+i] = simpleParams[6]
    newParams[6+len(consts[3]):] = simpleParams[7:] #copy remaining params
        
    newParams = opt.minimize(errFuncB1, newParams, (consts, normalWeight, slopeWeight, wDecay, skip, I), method=method)['x']
    newError = errFuncB1(newParams, consts, normalWeight, slopeWeight, wDecay, skip, I)

    print(i, "Error: ", newError)

    return newParams

def simFuncB1(params, consts, giveA=False): #option to return A and I

    dayNum = consts[0]
    measures = consts[3]
    
    A = np.zeros((dayNum))
    I = np.zeros((dayNum))
    
    A[0] = params[0]
    I[0] = params[1]
    
    gamma0 = params[2]
    gamma1 = params[3]
    nu = params[4]
    
    b0 = params[5]
    b1 = params[6:6+len(measures)]
    b2 = params[-1]
    
    currMeasure = 0
    currB1 = params[6+currMeasure] #th+e starting beta

    #iterate the arrays using the definition K' and I'
    for t in range(len(I)-1): #define I and K on range [1, length)
        
        while(t >= measures[currMeasure]): #we just passed a measure date
            currMeasure = currMeasure + 1
            currB1 = b1[currMeasure]
        
        diffA = (b0/ (1 + (currB1*I[t])**(b2) ) )*A[t] - gamma0*A[t]
        diffI = gamma1*A[t] - nu*I[t]
        
        A[t+1] = diffA + A[t]
        I[t+1] = diffI + I[t]

    if(giveA):
        return A,I
    return I
    
#x is the starting params, args = (consts, y)
def errFuncB1(params, consts, normalWeight, slopeWeight, wDecay, skip, y):

    x = simFuncB1(params, consts)
    
    error = 0
    if(normalWeight!=0):
        for t in range(skip,len(y)):
            error = error + ((y[t] - x[t])**2)*wDecay**(len(y)-t+1) #squared error
        error = error / len(y) # / T, average error
    
    slopeError = 0
    if(slopeWeight!=0):
        dy = np.diff(y)
        dx = np.diff(x)
        for t in range(skip,len(dy)):
            slopeError = slopeError +  ((dy[t] - dx[t])**2)*wDecay**(len(dy)-t+1) #squared error
        slopeError = slopeError / len(dy) # / T, average error

    return error*normalWeight + slopeError*slopeWeight




def startFuncB1(consts):
    
    params = np.zeros(7+len(consts[3]))
    params[0] = .001 + consts[1] #A(0), between 0 and 10 percent of the population
    params[1] = consts[1] #10e-10 + random.random()*.01 #I(0), between 0 and .1 percent of the population, avoid 0
    params[2] = .1 + random.random()*.4 #gamma0, asympt->recovs + infected - floor asympt infect rate
    params[3] = .001 + random.random()*.1 #gamma1, asympt->infected
    params[4] = .001 + random.random()*.1 #nu, infected -> recov or dead
    params[5] = .3 + random.random()*1.5 #beta0, ceiling infectRate - floor infect rate
    for i in range(len(consts[3])):
        params[6+i] = 13 + 125*random.random() #beta1, scaler on feedback
    params[6+len(consts[3])] = .5 + random.random()*2 #beta2, exponential [.5, 5.5]
    
    return params

def getParamsB1(I, consts, normalWeight=1, slopeWeight=0, wDecay=1, skip=0, randomIterCount=100, method="SLSQP"):
    bestParams = startFuncB1(consts)
    bestError = 10e10 #arbitrary large value
    for i in range(randomIterCount):
        print("Iter: ", i, end="")
        newParams = startFuncB1(consts)
        newParams = opt.minimize(errFuncB1, newParams, (consts, normalWeight, slopeWeight, wDecay, skip, I), method=method)['x']
        newError = errFuncB1(newParams, consts, normalWeight, slopeWeight, wDecay, skip, I)
    
        print("\r               \r", end="") #go back to the start of the line and write over
        if(newError < bestError):
            bestError = newError
            bestParams = newParams
            print(i, "New best error: ", bestError)

    return bestParams


##################################################### END ### Functions for fitting with multiple b1s



##################################################### Functions for fitting with multiple b2s

def refineParamsB2(I, simpleParams, consts, normalWeight=1, slopeWeight=0, wDecay=1, skip=0, method="SLSQP"):
    newParams = np.zeros(len(simpleParams)+len(consts[3]))
    
    newParams[0:7] = simpleParams[0:7] #copy from A(0) to b1
    
    for i in range(len(consts[3])): #copy b2 over to the multiple b2's
        newParams[7+i] = simpleParams[7]
        
    newParams = opt.minimize(errFuncB2, newParams, (consts, normalWeight, slopeWeight, wDecay, skip, I), method=method)['x']
    newError = errFuncB2(newParams, consts, normalWeight, slopeWeight, wDecay, skip, I)

    print(i, "Error: ", newError)

    return newParams

def simFuncB2(params, consts, giveA=False): #option to return A and I

    dayNum = consts[0]
    measures = consts[3]
    
    A = np.zeros((dayNum))
    I = np.zeros((dayNum))
    
    A[0] = params[0]
    I[0] = params[1]
    
    gamma0 = params[2]
    gamma1 = params[3]
    nu = params[4]
    
    b0 = params[5]
    b1 = params[6]
    b2 = params[7:7+len(measures)]
    
    currMeasure = 0
    currB2 = params[7+currMeasure] #th+e starting beta

    #iterate the arrays using the definition K' and I'
    for t in range(len(I)-1): #define I and K on range [1, length)
        
        while(t >= measures[currMeasure]): #we just passed a measure date
            currMeasure = currMeasure + 1
            currB2 = b2[currMeasure]
        
        diffA = (b0/ (1 + (b1*I[t])**(currB2) ) )*A[t] - gamma0*A[t]
        diffI = gamma1*A[t] - nu*I[t]
        
        A[t+1] = diffA + A[t]
        I[t+1] = diffI + I[t]

    if(giveA):
        return A,I
    return I
    
#x is the starting params, args = (consts, y)
def errFuncB2(params, consts, normalWeight, slopeWeight, wDecay, lastDay, skip, y):

    x = simFuncB2(params, consts)
    
    error = 0
    if(normalWeight!=0):
        for t in range(skip,len(y)):
            error = error + ((y[t] - x[t])**2)*wDecay**(len(y)-t+1) #squared error
        error = error / len(y) # / T, average error
        
        error = error + lastDay * ((y[-1] - x[-1])**2) #add error of the last day
    
    slopeError = 0
    if(slopeWeight!=0):
        dy = np.diff(y)
        dx = np.diff(x)
        for t in range(skip,len(dy)):
            slopeError = slopeError +  ((dy[t] - dx[t])**2)*wDecay**(len(dy)-t+1) #squared error
        slopeError = slopeError / len(dy) # / T, average error
        
        slopeError = slopeError + lastDay * ((dy[-1] - dx[-1])**2) #add error of the last day

    return error*normalWeight + slopeError*slopeWeight



def startFuncB2(consts):
    
    params = np.zeros(7+len(consts[3]))
    params[0] = .001 + consts[1] #A(0), between 0 and 10 percent of the population
    params[1] = consts[1] #10e-10 + random.random()*.01 #I(0), between 0 and .1 percent of the population, avoid 0
    params[2] = .1 + random.random()*.4 #gamma0, asympt->recovs + infected - floor asympt infect rate
    params[3] = .001 + random.random()*.1 #gamma1, asympt->infected
    params[4] = .001 + random.random()*.1 #nu, infected -> recov or dead
    params[5] = .3 + random.random()*1.5 #beta0, ceiling infectRate - floor infect rate
    params[6] = 13 + 125*random.random() #beta2, scaler on feedback
    for i in range(len(consts[3])):
        params[7+i] = .5 + random.random()*2 #beta2, exponential [.5, 5.5]
    
    return params

def getParamsB2(I, consts, normalWeight=1, slopeWeight=0, wDecay=1, lastDay=0, skip=0, randomIterCount=100, method="SLSQP"):
    bestParams = startFuncB2(consts)
    bestError = 10e10 #arbitrary large value
    for i in range(randomIterCount):
        print("Iter: ", i, end="")
        newParams = startFuncB2(consts)
        newParams = opt.minimize(errFuncB2, newParams, (consts, normalWeight, slopeWeight, wDecay, lastDay, skip, I), method=method)['x']
        newError = errFuncB2(newParams, consts, normalWeight, slopeWeight, wDecay, lastDay, skip, I)
    
        print("\r               \r", end="") #go back to the start of the line and write over
        if(newError < bestError):
            bestError = newError
            bestParams = newParams
            print(i, "New best error: ", bestError)
            
    return bestParams

##################################################### END ### Functions for fitting with multiple b1s


##################################################### Functions for fitting with multiple gamma0's

def refineParamsG0(I, simpleParams, consts, normalWeight=1, slopeWeight=0, wDecay=1, skip=0, method="SLSQP"):
    newParams = np.zeros(len(simpleParams)+len(consts[3]))
    
    newParams[0:7] = simpleParams[0:7] #copy from A(0) to b1
    
    for i in range(len(consts[3])): #copy b2 over to the multiple b2's
        newParams[7+i] = simpleParams[7]
        
    newParams = opt.minimize(errFuncG0, newParams, (consts, normalWeight, slopeWeight, wDecay, skip, I), method=method)['x']
    newError = errFuncG0(newParams, consts, normalWeight, slopeWeight, wDecay, skip, I)

    print(i, "Error: ", newError)

    return newParams

def simFuncG0(params, consts, giveA=False): #option to return A and I

    dayNum = consts[0]
    measures = consts[3]
    
    A = np.zeros((dayNum))
    I = np.zeros((dayNum))
    
    A[0] = params[0]
    I[0] = params[1]
    
    gamma0 = params[2:2+len(measures)]
    gamma1 = params[2+len(measures)]
    nu = params[3+len(measures)]
    
    b0 = params[4+len(measures)]
    b1 = params[5+len(measures)]
    b2 = params[6+len(measures)]
    
    currMeasure = 0
    currG0 = params[7+currMeasure] #th+e starting beta

    #iterate the arrays using the definition K' and I'
    for t in range(len(I)-1): #define I and K on range [1, length)
        
        while(t >= measures[currMeasure]): #we just passed a measure date
            currMeasure = currMeasure + 1
            currG0 = gamma0[currMeasure]
        
        diffA = (b0/ (1 + (b1*I[t])**(b2) ) )*A[t] - currG0*A[t]
        diffI = gamma1*A[t] - nu*I[t]
        
        A[t+1] = diffA + A[t]
        I[t+1] = diffI + I[t]

    if(giveA):
        return A,I
    return I
    
#x is the starting params, args = (consts, y)
def errFuncG0(params, consts, normalWeight, slopeWeight, wDecay, skip, y):

    x = simFuncG0(params, consts)
    
    error = 0
    if(normalWeight!=0):
        for t in range(skip,len(y)):
            error = error + ((y[t] - x[t])**2)*wDecay**(len(y)-t+1) #squared error
        error = error / len(y) # / T, average error
    
    slopeError = 0
    if(slopeWeight!=0):
        dy = np.diff(y)
        dx = np.diff(x)
        for t in range(skip,len(dy)):
            slopeError = slopeError +  ((dy[t] - dx[t])**2)*wDecay**(len(dy)-t+1) #squared error
        slopeError = slopeError / len(dy) # / T, average error

    return error*normalWeight + slopeError*slopeWeight



def startFuncG0(consts):
    
    params = np.zeros(7+len(consts[3]))
    params[0] = .001 + consts[1] #A(0), between 0 and 10 percent of the population
    params[1] = consts[1] #10e-10 + random.random()*.01 #I(0), between 0 and .1 percent of the population, avoid 0
    for i in range(len(consts[3])):
        params[2+i] = .1 + random.random()*.4 #gamma0, asympt->recovs + infected - floor asympt infect rate
    params[2+len(consts[3])] = .001 + random.random()*.1 #gamma1, asympt->infected
    params[3+len(consts[3])] = .001 + random.random()*.1 #nu, infected -> recov or dead
    params[4+len(consts[3])] = .3 + random.random()*1.5 #beta0, ceiling infectRate - floor infect rate
    params[5+len(consts[3])] = 13 + 125*random.random() #beta2, scaler on feedback
    params[6+len(consts[3])] = .5 + random.random()*2 #beta2, exponential [.5, 5.5]
    
    return params

def getParamsG0(I, consts, normalWeight=1, slopeWeight=0, wDecay=1, skip=0, randomIterCount=100, method="SLSQP"):
    bestParams = startFuncG0(consts)
    bestError = 10e10 #arbitrary large value
    for i in range(randomIterCount):
        print("Iter: ", i, end="")
        newParams = startFuncG0(consts)
        newParams = opt.minimize(errFuncG0, newParams, (consts, normalWeight, slopeWeight, wDecay, skip, I), method=method)['x']
        newError = errFuncG0(newParams, consts, normalWeight, slopeWeight, wDecay, skip, I)
    
        print("\r               \r", end="") #go back to the start of the line and write over
        if(newError < bestError):
            bestError = newError
            bestParams = newParams
            print(i, "New best error: ", bestError)
            
    return bestParams

##################################################### END ### Functions for fitting with multiple gamma0's


##################################################### Functions for fitting with constant parameters

def simFuncConst(params, consts):

    dayNum = consts[0]
    
    I = np.zeros((dayNum))
    I[0] = params[0]
    
    gamma = params[1]
    beta = params[2]
    #iterate the arrays using the definition K' and I'
    for t in range(len(I)-1): #define I and K on range [1, length)
        
        diffI = beta*I[t] - gamma*I[t]
        I[t+1] = diffI + I[t]
    return I
    
#x is the starting params, args = (consts, y)
def errFuncConst(params, consts, normalWeight, slopeWeight, wDecay, skip, y):

    x = simFuncConst(params, consts)
    
    error = 0
    if(normalWeight!=0):
        for t in range(skip,len(y)):
            error = error + ((y[t] - x[t])**2)*wDecay**(len(y)-t+1) #squared error
        error = error / len(y) # / T, average error
    
    slopeError = 0
    if(slopeWeight!=0):
        dy = np.diff(y)
        dx = np.diff(x)
        for t in range(skip,len(dy)):
            slopeError = slopeError +  ((dy[t] - dx[t])**2)*wDecay**(len(dy)-t+1) #squared error
        slopeError = slopeError / len(dy) # / T, average error

    return error*normalWeight + slopeError*slopeWeight


#param list is I(0), gamma, beta
#I' = beta*I - gamma*I
def startFuncConst(consts):
    
    params = np.zeros(3)
    params[0] = consts[1] #10e-10 + random.random()*.01 #I(0), between 0 and .1 percent of the population, avoid 0
    params[1] = .1 + random.random()*.4 #gamma, infect to recov+dead
    params[2] = .2 + random.random()*.4 #beta0, ceiling infectRate - floor infect rate
    
    return params

def getParamsConst(I, consts, normalWeight=1, slopeWeight=0, wDecay=1, skip=0, randomIterCount=100, method="SLSQP"):
    bestParams = startFuncConst(consts)
    bestError = 10e10 #arbitrary large value
    for i in range(randomIterCount):
        print("Iter: ", i, end="")
        newParams = startFuncConst(consts)
        newParams = opt.minimize(errFuncConst, newParams, (consts, normalWeight, slopeWeight, wDecay, skip, I), method=method)['x']
        newError = errFuncConst(newParams, consts, normalWeight, slopeWeight, wDecay, skip, I)
    
        print("\r               \r", end="") #go back to the start of the line and write over
        if(newError < bestError):
            bestError = newError
            bestParams = newParams
            print(i, "New best error: ", bestError)
            
    return bestParams

##################################################### END ### Functions for fitting with constant parameters




#fit last parameters

#consts = [[normal params], kappa1, kappa2, kappa3, beta1, beta2, beta3]
#params = [A(0), I(0)]


def startFuncEnd(consts):
    params = np.zeros(8)
    params[0] = .001 + consts[0][1] #A(0), between 0 and 10 percent of the population
    params[1] = consts[0][1] #10e-10 + random.random()*.01 #I(0), between 0 and .1 percent of the population, avoid 0
    return params

def getParamsEnd(I, consts, normalWeight=1, slopeWeight=0, wDecay=1, lastDay=0, skip=0, randomIterCount=100, method="SLSQP"):
    bestParams = startFuncEnd(consts)
    bestError = 10e10 #arbitrary large value
    for i in range(randomIterCount):
        print("Iter: ", i, end="")
        newParams = startFuncEnd(consts)
        newParams = opt.minimize(errFuncEnd, newParams, (consts, normalWeight, slopeWeight, wDecay, lastDay, skip, I), method=method)['x']
        newError = errFuncEnd(newParams, consts, normalWeight, slopeWeight, wDecay, lastDay, skip, I)
    
        print("\r               \r", end="") #go back to the start of the line and write over
        if(newError < bestError):
            bestError = newError
            bestParams = newParams
            print(i, "New best error: ", bestError)
            
    return bestParams


def simFuncEnd(params, consts, giveA=False): #option to return A and  I
        dayNum = consts[0][0]
        A = np.zeros((dayNum))
        I = np.zeros((dayNum))

        A[0] = params[0]
        I[0] = params[1]

        gamma0 = consts[1]
        gamma1 = consts[2]
        nu = consts[3]

        beta = consts[4:]

        #print(params)

        #iterate the arrays using the definition K' and I'
        for t in range(len(I)-1): #define I and K on range [1, length)
            diffA = (beta[0]/ (1 + (beta[1]*I[t])**(beta[2]) ) )*A[t] - gamma0*A[t]
            diffI = gamma1*A[t] - nu*I[t]

            A[t+1] = diffA + A[t]
            I[t+1] = diffI + I[t]

        if(giveA):
            return A,I

        return I #I is the data generated
    
   #x is the starting params, args = (consts, y)
def errFuncEnd(params, consts, normalWeight, slopeWeight, wDecay, lastDay, skip, y):
    x = simFuncEnd(params, consts)
    
    error = 0
    if(normalWeight!=0):
        for t in range(skip,len(y)):
            error = error + ((y[t] - x[t])**2)*wDecay**(len(y)-t+1) #squared error
        error = error / len(y) # / T, average error
        
        error = error + lastDay * ((y[-1] - x[-1])**2) #add error of the last day
    
    
    slopeError = 0
    if(slopeWeight!=0):
        dy = np.diff(y)
        dx = np.diff(x)
        for t in range(skip,len(dy)):
            slopeError = slopeError +  ((dy[t] - dx[t])**2)*wDecay**(len(dy)-t+1) #squared error
        slopeError = slopeError / len(dy) # / T, average error
        
        slopeError = slopeError + lastDay * ((dy[-1] - dx[-1])**2) #add error of the last day

    return error*normalWeight + slopeError*slopeWeight
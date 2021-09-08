import numpy as np
import matplotlib.pyplot as plt
import random

#generalized gradient descent class
#user must supply a random start function, simulate function, and check ranges function

#random start function:
#sim function: (params, const), return dataGen
#range function: (params, const), return params (bounded)
#random start function: (const), return params (random start)

#theta are variables, cons are constants

class GradDescent:
    
    #y: data to be approximated
    #x: simulated data that approximates x
    y = np.zeros(0) #empty numpy array until set
    x = np.zeros(0) #empty until set
    
    params = np.zeros(0) #vars
    consts = [] #constants
    
    delta = .001 #how much to move the variable (fraction based, non constant)
    eta = .001 #the learning rate
    slopeWeight = 0 #how heavily to weight the slope in the error function 
    
    velDecay = .9 #how much of the velocity to retain every iteration (i.e. .9 is 90%
    
    #functions defined during initilization
    begin = None
    sim = None
    constrainRanges = None
    
    #########################################################################################
    
    def __init__(self, data, consts, startFunc, simFunc, constrainFunc, params=None):
        self.y = data
        self.consts = consts
        
        if params is None:
            self.params == params
            
        self.begin = startFunc
        self.sim = simFunc
        self.constrainRanges = constrainFunc
    
    
    #########################################################################################
    
    
    def iterateVars(self, params, vel):
        
        x = self.simulate(params) #f(params)
        #get the gradient of f(params)
        gradient = np.zeros((len(params), len(self.y))) #f'(theta)
        for i in range(len(params)): #find the gradient for each var (partial deriv)
            paramsCopy = np.copy(params)
            
            varChange = paramsCopy[i] * self.delta #move some percent of the variable, this could also just be a constant 
            paramsCopy[i] = paramsCopy[i] + varChange # theta = theta + dtheta
            
            #print(varChange)
            gradient[i] = (self.simulate(paramsCopy) - x) / (varChange) #dy/dx essentially
        
        
        dy = np.diff(self.y) #slope of actual data
        dx = np.diff(x) #slope of generated data
        #get the gradient of f'(params)
        gradientSlope = np.zeros((len(params), len(dy))) #f'(theta)
        if(self.slopeWeight != 0):
            for i in range(len(params)):
                paramsCopy = np.copy(params)

                varChange = self.delta #paramsCopy[i] * self.delta #move some percent of the variable, this could also just be a constant 
                paramsCopy[i] = paramsCopy[i] + varChange # theta = theta + dtheta

                gradientSlope[i] = (np.diff(self.simulate(paramsCopy)) - dx) / (varChange) #dy/dx essentially
            
        
        #do calculus to solve this
        gradientChange1 = (1/len(self.y)) * gradient @ (-2*(self.y - x)) #normal gradient
        gradientChange2 = (1/len(dy)) * self.slopeWeight*gradientSlope @ (-2*(dy - dx)) #slope gradient
        paramsChange = self.eta * (gradientChange1 + gradientChange2)
        
        vel = vel*self.velDecay + paramsChange #update the velocity
        params = params - vel  #update parameters based on the velocity vector
        
        #print("pre constrain:", params)
        
        params = self.constrain(params) #verify variables are in bounds
        
        #print("post constrain:", params)
        
        return params, vel
    
    
    ##########################################################################################

    
    def getError(self): #squared error, can be modified to weight decay easily
        
        error = 0
        for t in range(len(self.y)):
            error = error + (self.y[t] - self.x[t])**2 #squared error
        error / len(self.y) # / T, average error
        
        
        slopeError = 0 #slope error
        if(self.slopeWeight != 0):
            dy = np.diff(self.y) #slope of actual
            dx = np.diff(self.x) #slope of generated
            for t in range(len(dy)):
                slopeError = slopeError + (dy[t] - dx[t])**2 #squared error
            slopeError / len(dy) # / T, average error
        
        
        return error + slopeError*self.slopeWeight #combine the two errors.

    
    #########################################################################################
    
    
    #printOut represents how many iterations to wait to update on solving progress
    def solveVars(self, printOut=0, params=None, maxIteration=10000000):
        
        if params is None:
            params = self.start()
        
        self.params = params
        self.x = self.simulate(params) #set x for the error calculation
        newError = self.getError()
        
        #print out the starting vars and error
        if(printOut != 0):
            print("Startin Vars:")
            print("Error:", newError)
            self.printVars(params)
        
        
        
        iteration = 0 #what iteration we're on
        vel = np.zeros(len(params)) #starting velocity of 0
        
        bestError = newError #the best vars configuration we've found so far
        lastImprovement = 0
        
        while(lastImprovement < 250 and iteration < maxIteration): #quit if no improvement for some iterations
            
            currentError = newError #progress currentError
            params, vel = self.iterateVars(params, vel)
        
            self.x = self.simulate(params)
            newError = self.getError()
        
            #print(iteration, params, newError)
        
            iteration = iteration + 1
            lastImprovement = lastImprovement + 1
        
            if(newError < bestError): #improvement on error
                
                #print new error
                if(printOut != 0 and iteration % printOut == 0):
                    change = (currentError - newError)/currentError
                    print("Iteration:", iteration, "Error:", newError, ", improvement:", change)
                    
                self.params = np.copy(params)
                bestError = newError
                lastImprovement = 0

        self.x = self.simulate(self.params) #use the best parametrs to simulate data

        return self.params
    
    
    #########################################################################################
    #########################################################################################
    
    def printVars(self, params):
        print(params)
        
    def simulate(self, params):
        return self.sim(params, self.consts)
        
    def constrain(self, params):
        return self.constrainRanges(params, self.consts)

    
    def start(self):
        return self.begin(self.consts)
        
    #########################################################################################
    #########################################################################################

    def graph(self):
        fig, ax = plt.subplots(figsize=(18,8))
        ax.plot(self.dataActual, color="red")
        ax.plot(self.dataGen, color="red", linestyle="dashed")
        
        return fig, ax
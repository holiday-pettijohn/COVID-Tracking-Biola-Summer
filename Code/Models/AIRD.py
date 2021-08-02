import numpy as np
import matplotlib.pyplot as plt
import random

#min: (y-f(theta))^2
#y is the infections, and f is the function to simulate infections given the parameters
#theta = [I(0), A(0), q, gamma0, gamma1, kappa, beta0, beta1, beta2, beta3]
# A' = (b0 + b1 / (1 + (b2*I)^b3 ) ) * (q*Pop * A) / (q*Pop + A) - kappa*A - gamma0*A
# I' = kappa*A - gamma1*I

#A' = A(t+1) - A(t)

#note that I should be modified so pop = 1
class AIRD:
    actualI = np.zeros(0) #empty numpy array until set
    theta = np.zeros(0) #the current variables, when solving this is set at the end
    I = np.zeros(0) #simulated infections
    A = np.zeros(0) #simulated asymptomatics
    
    def setInfections(self, infect, pop = 1): #define the infections for the model, if pop is set, infections will be scaled
        self.actualI = infect / pop

    #simulate the number of infections given the variables theta
    def simulate(self, theta, setSelf=False):
        I = np.zeros(np.shape(self.actualI))
        A = np.zeros(np.shape(self.actualI))

        I[0] = theta[0] # I(0)
        A[0] = theta[1] # A(0)
        q = theta[2]
        gamma0 = theta[3]
        gamma1 = theta[4]
        kappa = theta[5]
        beta = theta[6:] #beta0, beta1, beta2, beta3

        #iterate the arrays using the definition A' and I'
        for t in range(len(I)-1): #define I and A on range [1, length)
            diffA = (beta[0] + (beta[1] / (1 + (beta[2]*I[t])**beta[3] )) ) * (q*A[t])/(q + A[t]) - kappa*A[t] - gamma0*A[t]
            diffI = kappa*A[t] - gamma1*I[t]

            A[t+1] = diffA + A[t]
            I[t+1] = diffI + I[t]
        
        if(setSelf):
            self.I = I
            self.A = A
        
        return I

    #delta is the percent to move the var, eta is the learning rate
    def iterateVars(self, theta, eta, delta): 

        simulatedI = self.simulate(theta) #f(theta)

        gradient = np.zeros((len(theta), len(self.actualI))) #f'(theta)
        for i in range(len(theta)): #find the gradient for each var (partial deriv)
            thetaCopy = np.copy(theta)
            
            #varChange = theta[i] * delta #move some percent of the variable, this could also just be a constant instead
            varChange = delta
            
            thetaCopy[i] = thetaCopy[i] + varChange
            
            gradient[i] = (self.simulate(thetaCopy) - simulatedI) / (varChange)
        
        thetaNew = theta - eta * gradient @ (-2*(self.actualI - simulatedI)) #formula for iterating the variables
        
        thetaNew = self.checkRanges(thetaNew) #verify variables are in bounds
        
        return thetaNew
    
    def getError(self): #squared error, can be modified to weight decay easily
        error = 0
        
        for t in range(len(self.actualI)):
            error = error + (self.actualI[t] - self.I[t])**2 #squared error
            
        error / len(self.actualI) # / T, average error
        
        return error
    
    def boundVariable(self, val, minimum, maximum):
        if(val < minimum):
            return minimum
        elif(val > maximum):
            return maximum
        else:
            return val
    
    def checkRanges(self, theta): #make sure all the variables are within range
        theta[0] = self.boundVariable(theta[0], 0, 1) #I(t)
        theta[1] = self.boundVariable(theta[1], 0.000001, 1) #A(t)
        
        theta[2] = self.boundVariable(theta[2], .00001, 1) #q
        
        theta[3] = self.boundVariable(theta[3], 0.00001, 1) #gamma0
        theta[4] = self.boundVariable(theta[4], 0.00001, 1) #gamma1
        
        theta[5] = self.boundVariable(theta[5], .00001, 1) #kappa
        
        theta[6] = self.boundVariable(theta[6], 0.00001, 5) #beta0
        theta[7] = self.boundVariable(theta[7], 0.00001, 5) #beta1
        theta[8] = self.boundVariable(theta[8], 0, 1e10) #beta2
        theta[9] = self.boundVariable(theta[9], 0, 100) #beta3
        
        return theta
        
            
        
    
    def getRandomInit(self): #random starting conditions
        theta = np.zeros(10) #10 vars to solve
        
        theta[0] = random.random()*.001 #I(0), between 0 and .1 percent of the population
        theta[1] = random.random()*.003 #A(0), between 0 and .3 percent of the population
        
        theta[2] = .01 + random.random()*.99 #q: [.01, 1]
        
        theta[3] = .001 + random.random()*.1 #gamma0 asympt recov rate [.001, .101]
        theta[4] = .001 + random.random()*.1 #gamma1 infect recov rate [.001, .101]
        
        theta[5] = .01 + random.random()*.5 #kappa, asypt -> infect rate [.01, .51]
        
        theta[6] = .001 + random.random()*.5 #beta0, floor infection rate [.001, .501]
        theta[7] = .001 + random.random()*.5 #beta1, floor infection rate [.001, .501]
        theta[8] = 1 + random.random()*500 #beta2, scaler to I [1, 501]
        theta[9] = .5 + random.random()*5 #exponential for I*beta2 [.5, 5.5]
        
        return theta
    
    def solveVars(self, eta=.001, delta=.01, printOut=0):
        theta = self.getRandomInit() #get a random starting positions for the variables
        
        if(printOut > 1):
            print("Starting vars:")
            self.printTheta(theta)
        
        self.simulate(theta, setSelf=True)
        newError = self.getError()
        
        iteration = 0
        change = 1 #init val doesn't matter as long as change >= .001
        while(abs(change) > .001 and iteration < 10000): #while we improve/deprove more than .01% (not converged)
            currentError = newError #progress currentError
            theta = self.iterateVars(theta, eta, delta)
            
            self.simulate(theta, setSelf=True)
            newError = self.getError()
            
            change = (currentError - newError)/currentError
            if(printOut > 0):
                print("Error:", newError, ", improvement:", change)
                if(printOut > 1):
                    self.printTheta(theta)
                    
            iteration = iteration + 1
        
        self.theta = theta #set variables since they converged
        #self.I and self.A will be set during the final simulate(theta)
        
        return theta
    
    def printTheta(self, theta):
        print("I(0), A(0):", theta[0], theta[1])
        print("q:", theta[2])
        print("gamma:", theta[3:5])
        print("kappa:", theta[5])
        print("beta:", theta[6:])
        print() #spacer
    
    def graph(self, graphA = False):
        fig, ax = plt.subplots(figsize=(18,8))
        ax.plot(self.actualI, color="red")
        ax.plot(self.I, color="red", linestyle="dashed")
        
        if(graphA):
            ax.plot(self.A, color="orange", linestyle="dashed")
        
        return fig, ax
        
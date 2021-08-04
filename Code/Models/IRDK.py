import numpy as np
import matplotlib.pyplot as plt
import random

#min: (y-f(theta))^2
#y is the infections, and f is the function to simulate infections given the parameters

#theta = [I(0), A(0), q, alpha0, alpha1, beta0, beta1, beta2]

# I' = (b0 + b1 / (1 + (K)^b2 ) ) * (q*Pop * I) / (q*Pop + I) - R' - D'
# K' = alpha0I - alpha1*K #known public perception of the virus
#K(0) = 0

#you don't need a scaler on K since it is arbitrary, this will e adjusted by alpha0

#note that I should be modified so pop = 1
class IRDK:
    actualI = np.zeros(0) #empty numpy array until set
    deathRate = 0
    recovRate = 0
    
    theta = np.zeros(0) #the current variables, when solving this is set at the end
    I = np.zeros(0) #simulated infections
    K = np.zeros(0) #simulated known
    
    
    def setRecovRate(self, I, R, D):
        #R' = gamma0*I
        #D' = gamma1*I
        
        y = np.zeros((len(I)-1,1))
        x = np.zeros((len(I)-1,1))

        # R(t+1) - R(t) = gamma*I(t)
        y[:,0] = R[1:] - R[:-1]
        x[:,0] = I[:-1]
        
        self.recovRate = np.linalg.lstsq(x, y, rcond = None)[0].flatten()[0] #solve for gamma
        
        # D(t+1) - D(t) = *I(t)
        y[:,0] = D[1:] - D[:-1]
        #x is already set correctly
        
        self.deathRate = np.linalg.lstsq(x, y, rcond = None)[0].flatten()[0] #solve for gamma
        
    
    def setInfections(self, I, pop = 1): #define the infections for the model, if pop is set, infections will be scaled
        self.actualI = I / pop

    #simulate the number of infections given the variables theta
    def simulate(self, theta, setSelf=False):
        I = np.zeros(np.shape(self.actualI))
        K = np.zeros(np.shape(self.actualI))

        I[0] = theta[0] # I(0)
        K[0] = theta[1]
        q = theta[2]
        alpha0 = theta[3]
        alpha1 = theta[4]
        beta = theta[5:] #beta0, beta1, beta2

        #iterate the arrays using the definition K' and I'
        for t in range(len(I)-1): #define I and K on range [1, length)
            diffI = (beta[0] + (beta[1] / (1 + (K[t])**beta[2] )) ) * (q*I[t])/(q + I[t]) - (self.recovRate + self.deathRate)*I[t]
            diffK = alpha0*I[t] - alpha1*K[t]

            I[t+1] = diffI + I[t]
            K[t+1] = diffK + K[t]
            
        if(setSelf):
            self.I = I
            self.K = K
        
        return I

    #delta is the percent to move the var, eta is the learning rate
    def iterateVars(self, theta, vel, velDecay,eta, delta, w): 

        simulatedI = self.simulate(theta) #f(theta)
        
        gradient = np.zeros((len(theta), len(self.actualI))) #f'(theta)
        for i in range(len(theta)): #find the gradient for each var (partial deriv)
            thetaCopy = np.copy(theta)
            
            #varChange = theta[i] * delta #move some percent of the variable, this could also just be a constant instead
            varChange = delta
            
            thetaCopy[i] = thetaCopy[i] + varChange
            
            gradient[i] = (self.simulate(thetaCopy) - simulatedI) / (varChange)
        
        
        actualChangeI = np.diff(self.actualI)
        simulatedChangeI = np.diff(simulatedI)
        gradientSlope = np.zeros((len(theta), len(actualChangeI))) #f'(theta)
        for i in range(len(theta)):
            thetaCopy = np.copy(theta)
            
            #varChange = theta[i] * delta #move some percent of the variable, this could also just be a constant instead
            varChange = delta
            
            thetaCopy[i] = thetaCopy[i] + varChange
            
            gradientSlope[i] = (np.diff(self.simulate(thetaCopy)) - simulatedChangeI) / (varChange)
        
        thetaChange0 = (1/len(self.actualI)) * gradient @ (-2*(self.actualI - simulatedI)) #normal gradient
        thetaChange1 = (1/len(actualChangeI)) * w*gradientSlope @ (-2*(actualChangeI - simulatedChangeI)) #slope gradient
        thetaChange = eta * (thetaChange0 + thetaChange1)
        
        vel = vel*velDecay + thetaChange #update the velocity
        theta = theta - vel  #update theta based on the velocity vector
        theta = self.checkRanges(theta) #verify variables are in bounds
        
        return theta, vel
    
    def getError(self, w): #squared error, can be modified to weight decay easily
        error = 0
        
        for t in range(len(self.actualI)):
            error = error + (self.actualI[t] - self.I[t])**2 #squared error
            
        error / len(self.actualI) # / T, average error
        
        
        slopeError = 0 #error on I'
        changeI = np.diff(self.actualI) #I'
        changeSimI = np.diff(self.I) #simulated I change
        for t in range(len(changeI)):
            slopeError = slopeError + (changeI[t] - changeSimI[t])**2 #squared error
        slopeError / len(changeI) # / T, average error
        
        error = error + slopeError*w #combine the two errors.
        
        return error
    
    def boundVariable(self, val, minimum, maximum):
        if(val < minimum):
            return minimum
        elif(val > maximum):
            return maximum
        else:
            return val
    
    def checkRanges(self, theta): #make sure all the variables are within range
        theta[0] = self.boundVariable(theta[0], 0, .005) #I(t)
        theta[1] = self.boundVariable(theta[1], 0.000001, 10e25) #K(t)
        
        theta[2] = self.boundVariable(theta[2], .00001, 1) #q
        
        theta[3] = self.boundVariable(theta[3], 0.00001, 10e10) #alpha0
        theta[4] = self.boundVariable(theta[4], 0.00001, 1) #alpha1
        
        theta[5] = self.boundVariable(theta[5], 0.0000001, self.recovRate + self.deathRate) #beta0 < recoveries
        #theta[5] = 0
        #theta[6] = self.boundVariable(theta[6], 0.00001, 5) #beta1
        theta[6] = self.boundVariable(theta[6], self.recovRate + self.deathRate - theta[6], 5)
        theta[7] = self.boundVariable(theta[7], .25, 10) #beta3
        
        return theta
        
            
        
    
    def getRandomInit(self): #random starting conditions
        theta = np.zeros(8) #8 vars to solve
        
        theta[0] = random.random()*.001 #I(0), between 0 and .1 percent of the population
        theta[1] = random.random()*.1 #A(0), between 0 and 10 percent of the population
        
        theta[2] = .1 + random.random()*.9 #q: [.1, 1]
        
        theta[3] = .1 + random.random()*10 #alpha0 knowledge rate [.001, 10.1]
        theta[4] = .001 + random.random()*.05 #alpha1 knowledge decay rate [.001, .051]
        
        theta[5] = (self.recovRate+self.deathRate) * random.random() #beta0, floor infection rate [.001, .201]
        #theta[5] = 0
        theta[6] = (self.recovRate+self.deathRate - theta[5]) * (.5*random.random() + 1) #beta1, ceiling trans rate, 1 to 1.5 range
        theta[7] = .5 + random.random()*5 #beta2, exponential [.5, 5.5]
        
        return theta
    
    def solveVars(self, velDecay = .9, eta=.001, delta=.01, w=1, printOut=0, theta=None):
        if(theta==None): #if not set use random
            theta = self.getRandomInit() #get a random starting positions for the variables
        
        if(printOut > 1):
            print("Starting vars:")
            self.printTheta(theta)
        
        self.simulate(theta, setSelf=True)
        newError = self.getError(w)
        
        vel = np.zeros(len(theta)) #starting velocity = 0
        change = 1 #init val doesn't matter as long as change >= .001
        
        bestError = newError
        
        iteration = 0
        lastImprovement = 0
        
        while(lastImprovement < 250): #quit if no improvement for some iterations
            currentError = newError #progress currentError
            theta, vel = self.iterateVars(theta, vel, velDecay, eta, delta, w)
            
            self.simulate(theta, setSelf=True)
            newError = self.getError(w)
                    
            if(newError < bestError): #new best theta
                if(printOut > 1):
                    change = (currentError - newError)/currentError
                    print("Iteration:", iteration, "Error:", newError, ", improvement:", change)
                self.theta = np.copy(theta) #update since error has improved
                bestError = newError
                lastImprovement = 0
            else: #no improvement
                lastImprovement = lastImprovement + 1
                
            iteration = iteration + 1
        
        self.simulate(self.theta, setSelf=True) #set I and K
        
        if(printOut > 0):
            self.printTheta(self.theta)
        
        return self.theta
    
    
    def printTheta(self, theta):
        print("I(0), A(0):", theta[0], theta[1])
        print("q:", theta[2])
        print("alpha:", theta[3:5])
        print("beta:", theta[5:])
        print() #spacer
    
    def graph(self, graphK = False):
        fig, ax = plt.subplots(figsize=(18,8))
        ax.plot(self.actualI, color="red")
        ax.plot(self.I, color="red", linestyle="dashed")
        
        if(graphK):
            scaleFactor = max(self.actualI) / max(self.K)
            print("K scaled by:", scaleFactor)
            ax.plot(self.K*scaleFactor, color="orange", linestyle="dashed")
     
        return fig, ax
        
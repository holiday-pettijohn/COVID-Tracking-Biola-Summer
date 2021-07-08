import Models.SIRD as sird
import Models.SIRD_Time as sird_time
import Models.SIRD_Beta_Time as sird_beta

import Models.SIRD_Feedback as sird_fb
import Models.SIRD_Feedback_Delay as sird_fd

import Models.process as process

import numpy as np
import csv
import matplotlib.pyplot as plt
import platform


def loadCalEarly():
    pathc="../Data/State Data/"
    if platform.system() == "Windows":
        pathc.replace("/", "\\")

    filename = "CA.csv"
    dates, infectRaw, recovRaw, deadRaw = process.loadData(pathc + filename)

    recovRaw = process.getRecov(infectRaw, deadRaw)
    infectRaw = process.getCurrentInfect(infectRaw, recovRaw, deadRaw)

    pop = 40000000
    
    skipDays = 0
    numDays = 275 #from start to end of first wave
    infect = infectRaw[skipDays:numDays]
    recov = recovRaw[skipDays:numDays] 
    dead = deadRaw[skipDays:numDays]
    
    return dates, infect, recov, dead, pop

def loadCalAll():
    pathc="../Data/State Data/"
    if platform.system() == "Windows":
        pathc.replace("/", "\\")

    filename = "CA.csv"
    dates, infectRaw, recovRaw, deadRaw = process.loadData(pathc + filename)

    recovRaw = process.getRecov(infectRaw, deadRaw)
    infectRaw = process.getCurrentInfect(infectRaw, recovRaw, deadRaw)

    pop = 40000000
    
    skipDays = 0
    numDays = len(infectRaw)
    infect = infectRaw[skipDays:numDays]
    recov = recovRaw[skipDays:numDays] 
    dead = deadRaw[skipDays:numDays]
    
    return dates, infect, recov, dead, pop
    
def loadItaEarly():
    pathc= "../Data/Italian Data/"
    if platform.system() == "Windows":
        pathc.replace("/", "\\")

    filename = "National Data.csv"
    dates, infectRaw, recovRaw, deadRaw = process.loadData(pathc + filename)

    pop = 60000000
    
    skipDays = 0
    numDays = 160 #from start to end of first wave
    infect = infectRaw[skipDays:numDays]
    recov = recovRaw[skipDays:numDays] 
    dead = deadRaw[skipDays:numDays]
    
    return dates, infect, recov, dead, pop
    
def loadItaAll():
    pathc= "../Data/Italian Data/"
    if platform.system() == "Windows":
        pathc.replace("/", "\\")

    filename = "National Data.csv"
    dates, infectRaw, recovRaw, deadRaw = process.loadData(pathc + filename)

    pop = 60000000
    
    skipDays = 0
    numDays = len(infectRaw)
    infect = infectRaw[skipDays:numDays]
    recov = recovRaw[skipDays:numDays] 
    dead = deadRaw[skipDays:numDays]
    
    return dates, infect, recov, dead, pop
    

#dates, infect, recov, dead, pop = loadItaEarly() #change function for different data
#dates, infect, recov, dead, pop = loadItaAll() #change function for different data
dates, infect, recov, dead, pop = loadCalEarly() #change function for different data
#dates, infect, recov, dead, pop = loadCalAll() #change function for different data

fig, ax = plt.subplots(figsize=(18,8))
ax.set_title("Transmission Rate (CA, All Time)", fontsize = 35)

lW = 6  #line thickness

fig2, ax2 = plt.subplots(figsize=(18,8))
ax2.set_title("Predictions (CA)", fontsize = 35)

#set up params
sird.weightDecay= .93
sird.regularizer=10

sird_fd.weightDecay = .93
sird_fd.regularizer = 10
sird_fd.betaUseDecay = True

sird_fd.delay = 21
#setup params


linVars = [ .06340781449212123 , .14704240670154564, 0.07617616075268935, 0.00121143972523787]
nonLinVars = [350.0, 4.999999999999999]
q = 0.30117300575

#linVars = [ -0.055354205755110006, 0.12255307448722035, 0.08555686020188996, 0.0010161885990735459]
#nonLinVars = [15.0, 5.0]
#q = 0.30117300575



#get q and suscept pop
#q = sird.getQ(infect,recov, dead, pop) #use non feedback model to get q value, should be accurate enough
print("q =", q)
print("Delay =", sird_fd.delay)
print("FB weight decay =", sird_fd.weightDecay)

#q=.011
suscept = process.getSuscept(infect,recov,dead, q,pop)
#get q and suscept pop



#grid and solve non lin vars
b1Range = (0, 100) #modify to get finer results
b2Range = (0, 5)
betaVarsResol = [50, 8]

#linVars, nonLinVars = sird_fd.solveAllVars(suscept, infect, recov, dead, [b1Range, b2Range], betaVarsResol, printOut=True)

#grid and solve non lin vars

#daysToPredict = 150


#plot
betaTime = sird_fd.getBetaTime(suscept, infect, recov, dead, linVars, nonLinVars)
linVarsTime = sird_time.getLinVars(suscept, infect, recov, dead)
linVarsConst = sird.getLinVars(suscept, infect, recov, dead)

ax.plot(linVarsTime[:,0], color="red", label="Actual", linewidth=lW*.4) #time varying beta
ax.plot(np.ones(len(linVarsTime[:,0]))*linVarsConst[0], color="blue", linestyle="dotted", label="Constant", linewidth=lW) #constant beta
ax.plot(betaTime, color="green", linestyle="dashed", label="Feedback", linewidth=lW) #feedback beta

#Customizing the Figure
ax.tick_params(axis="both", labelsize=20)

ax.set_xlabel("Time", fontsize = 30)
ax.set_ylabel("β", fontsize = 30)

ax.legend(fontsize = 30)
ax.set_ylim([0,.25])

#plot


#plot2
dTP = len(suscept)-160

linVarsConst2 = sird.getLinVars(suscept[:-dTP], infect[:-dTP], recov[:-dTP], dead[:-dTP]) #use this on initial spike

sp, ipConst, rp, dp = sird.predictMatch(suscept, infect, recov, dead, dTP, linVars=linVarsConst, graph=False)
sp, ipFeed, rp, dp = sird_fd.predictMatch(suscept, infect, recov, dead, dTP, linVars=linVars, nonLinVars=nonLinVars, graph=False)

ax2.plot(infect/1000, color="red", label="Actual", linewidth=lW)
ax2.plot(ipConst/1000, color="blue", label="Constant", linestyle="dotted", linewidth=lW)
ax2.plot(ipFeed/1000, color="green", label="Feedback", linestyle="dashed", linewidth=lW)

ax2.axvline(len(suscept)-dTP, color='black', linestyle='dotted', linewidth=lW)

#Customizing the Figure
ax2.tick_params(axis="both", labelsize=20)

ax2.set_xlabel("Time", fontsize = 30)
ax2.set_ylabel("Infected (in Thousands)", fontsize = 30)

ax2.set_ylim([0, max(infect)*1.2/1000])

ax2.legend(fontsize = 30, loc='upper left')
#plot



#params for early Italy:
#q = 0.2431392053333333
#Delay = 21
#Solution:
#b0:  0.00872621927978879
#b1:  0.177559789522771
#g:   0.030023497085678866
#nu:  0.003125651452747975
#b2:  4850.0
#b3:  1.0
#linVars = [ 0.00872621927978879, 0.177559789522771, 0.030023497085678866, 0.003125651452747975]
#nonLinVars = [4850.0, 1.0]
#q = 0.2431392053333333

#params for all Italy:
#q = 0.2846452188333334
#Delay = 21
#Solution:
#b0:  0.02773790203107795
#b1:  0.08361680058861755
#g:   0.03666749736275282
#nu:  0.0006749992291412268
#b2:  155.0
#b3:  5.0
#cost:  12858397.275609879
#linVars = [ 0.02773790203107795, 0.08361680058861755, 0.03666749736275282, 0.0006749992291412268]
#nonLinVars = [155.0, 5.0]
#q = 0.2846452188333334

#params for early California:
#q = 0.25729957600000003
#Delay = 21
#FB weight decay = 0.94
#Solution:
#b0:  0.06340781449212123
#b1:  0.14704240670154564
#g:   0.07617616075268935
#nu:  0.00121143972523787
#b2:  350.0
#b3:  4.999999999999999
#cost:  131432.9974546541
#linVars = [ .06340781449212123 , .14704240670154564, 0.07617616075268935, 0.00121143972523787]
#nonLinVars = [350.0, 4.999999999999999]
#q = 0.30117300575

#params for all California:
#q = 0.30117300575
#Delay = 21
#FB weight decay = 0.93
#Solution:
#b0:  -0.026942664031480594
#b1:  0.09272694800909409
#g:   0.08541626656698537
#nu:  0.0008505863398209726
#b2:  16.0
#b3:  5.0
#cost:  9296.355103419619
#linVars = [ -0.026942664031480594, 0.09272694800909409, 0.08541626656698537, 0.0008505863398209726]
#nonLinVars = [16.0, 5.0]
#q = 0.30117300575

def getFitError(I, IP):
    return sum((I - IP)**2) #squared error

#gridding functions
def getFitLin(suscept, infect, recov, dead, linVars, initDays):
    resol = 50 #test 25 values above 1 and 25 values below
    
    minError = 1e100 #let this be an starting enormous error
    minVal = 0
    
    for i in range(1, resol+1): #test vals below 1
        infectTest = infect.copy()
        infectTest[0:initDays] = infectTest[0:initDays] *(.5 + 1.5*( i / resol)) #basically scale between .5 and 2
        
        sp,ip,rp,dp = sird.predictMatch(suscept, infectTest, recov, dead, len(suscept)-initDays, linVars=linVars, graph=False)
        currError = getFitError(infect, ip)
        
        if(currError < minError):
            minError = currError
            minVal = (.5 + 1.5*( i / resol))
        
    print("Scale by:", minVal)
    
    
    #calculate min I
    infectTest = infect.copy()
    infectTest[0:initDays] = infectTest[0:initDays] * minVal #scaler
    sp,ip,rp,dp = sird.predictMatch(suscept, infectTest, recov, dead, len(suscept)-initDays, linVars=linVars, graph=False)
    
    return ip

def getFitFeed(suscept, infect, recov, dead, linVars, nonLinVars, initDays):
    resol = 50
    
    minError = 1e100 #let this be an starting enormous error
    minVal = 0
    for i in range(1,resol+1):
        infectTest = infect.copy()
        for j in range(0, initDays):
            infectTest[j] = infectTest[j] * (.7 + .3*i/resol)**(initDays - j) #weight decay
            
        sp,ip,rp,dp = sird_fd.predictMatch(suscept, infectTest, recov, dead, len(suscept)-initDays, linVars=linVars, nonLinVars=nonLinVars, graph=False)
        currError = getFitError(infect, ip)
            
        if(currError < minError):
            minError = currError
            minVal = (.7 + .3*i/resol)
    
    print("Weight by:", minVal)
    
    #calculate min I
    infectTest = infect.copy()
    for j in range(0, initDays):
        infectTest[j] = infectTest[j] * (minVal)**(initDays - j) #weight decay
    sp,ip,rp,dp = sird_fd.predictMatch(suscept, infectTest, recov, dead, len(suscept)-initDays, linVars=linVars, nonLinVars=nonLinVars, graph=False)

    return ip
#gridding functions





fig3, ax3 = plt.subplots(figsize=(18,8))
ax3.set_title("Initial Conditions and Fit (CA)", fontsize = 35)

ipFeed = getFitFeed(suscept, infect, recov, dead, linVars, nonLinVars, len(suscept)-dTP)
ipConst = getFitLin(suscept, infect, recov, dead, linVarsConst, len(suscept)-dTP)

ax3.plot(infect/1000, color="red", label="Actual", linewidth=lW)
ax3.plot(ipConst/1000, color="blue", label="Constant", linestyle="dotted", linewidth=lW)
ax3.plot(ipFeed/1000, color="green", label="Feedback", linestyle="dashed", linewidth=lW)


ax3.axvline(len(suscept)-dTP, color='black', linestyle='dotted', linewidth=lW)

#Customizing the Figure
ax3.tick_params(axis="both", labelsize=20)

ax3.set_xlabel("Time", fontsize = 30)
ax3.set_ylabel("Infected (in Thousands)", fontsize = 30)

ax3.legend(fontsize = 30, loc='upper left')

ax3.set_ylim([0, max(infect)*1.2/1000])

#ax2.set_xlim(100)
#ax3.set_xlim(100)

plt.show()


process.writeCSV(ax, dates, "..\\Figures\\CSVs\\betaCalEarly.csv")
process.writeCSV(ax2, dates, "..\\Figures\\CSVs\\predictCalEarly.csv")
process.writeCSV(ax3, dates, "..\\Figures\\CSVs\\fitCalEarly.csv")
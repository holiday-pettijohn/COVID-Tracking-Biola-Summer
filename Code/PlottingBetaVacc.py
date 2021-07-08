import Models.SIRD as sird
import Models.SIRD_Time as sird_time
import Models.SIRD_Beta_Time as sird_beta

import Models.SIRD_Feedback as sird_fb
import Models.SIRD_Feedback_Delay as sird_fd
import Models.SIRDV as sirdv

import Models.SIRDV_Time as sirdv_time
import Models.SIRDV_Feedback_Delay as sirdv_fd

import Models.process as process

import numpy as np
import csv
import matplotlib.pyplot as plt
import platform



def loadCalAllVacc(datesN, iN, rN, dN): #take the dates from a normal file and make sure start dates align data should be the same before V
    pathc="../Data/Vaccination Data/State Data With Vaccinations/"
    if platform.system() == "Windows":
        pathc.replace("/", "\\")

    filename = "CA.csv"
    dates, I, R, D, vacc = sirdv.loadData(pathc + filename)
    
    skipDays = 0
    numDays = len(I)
    dates = dates[skipDays:numDays]
    
    firstVaccIndex = 0
    while(vacc[firstVaccIndex] == 0):
        firstVaccIndex += 1
        
    firstVaccDate = dates[firstVaccIndex]
    
    #match in other dates
    matchDate = 0
    while(datesN[matchDate] != firstVaccDate):
        matchDate += 1
    
    endDate = datesN[-1]
    endIndex = len(dates) - 1
    while(dates[endIndex] != endDate):
        endIndex += -1
    
    #now insert vacc data properly
    vaccNew = np.zeros(len(datesN))
    
    vaccNew[matchDate:] = vacc[firstVaccIndex: endIndex+1]
    
    return vaccNew

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
    dates = dates[skipDays:numDays]
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
    dates = dates[skipDays:numDays]
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
    dates = dates[skipDays:numDays]
    infect = infectRaw[skipDays:numDays]
    recov = recovRaw[skipDays:numDays] 
    dead = deadRaw[skipDays:numDays]
    
    return dates, infect, recov, dead, pop





#dates, infect, recov, dead, pop = loadItaEarly() #change function for different data
#dates, infect, recov, dead, pop = loadItaAll() #change function for different data
#dates, infect, recov, dead, pop = loadCalEarly() #change function for different data
dates, infect, recov, dead, pop = loadCalAll() #change function for different data

lW=6

vacc = loadCalAllVacc(dates, infect, recov, dead)

fig, ax = plt.subplots(figsize=(18,8))
ax.set_title("Transmission Rate (California, All Time)", fontsize = 35)

fig2, ax2 = plt.subplots(figsize=(18,8))
ax2.set_title("Predictions (California)", fontsize = 35)

#set up params
sird.regularizer=10

sird.weightDecay= .94
sirdv.weightDecay = .94
sirdv_fd.weightDecay = .94
sird_fd.weightDecay = .94

sird_fd.regularizer = 10
sirdv_fd.regularizer = 10

sirdv_fd.betaUseDecay = True
sird_fd.betaUseDecay = True

sird_fd.delay = 14
sirdv_fd.delay = 14
#setup params

#get q and suscept pop
#q = sird.getQ(infect,recov, dead, pop) #use non feedback model to get q value, should be accurate enough

linVarsV = [ 0.061108706122272954, 0.005273168933039929, 0.08520091987532212, 0.0010161885990735459]
nonLinVarsV = [130.0, 4.999999999999999]
q = 0.30117300575

print("q =", q)
print("Delay =", sird_fd.delay)
print("FB weight decay =", sird_fd.weightDecay)
print("FBV weight decay =", sirdv_fd.weightDecay)

vacc = vacc*q #scale down

#q=.011
suscept = process.getSuscept(infect,recov,dead, q,pop)

susceptV, recovV = sirdv.approxSusceptRecov(suscept.copy(), recov.copy(), vacc)


fig5, ax5 = plt.subplots(figsize=(18,8))

ax5.plot(vacc/1000, label="Vaccinated", color="green", linewidth=6)
ax5.plot(susceptV/1000, label="Susceptible", color="purple", linewidth=6)
ax5.plot(recovV/1000, label="Recovered", color="blue", linewidth=6)
ax5.plot(dead/1000, label="Dead", color="black", linewidth=6)
ax5.plot(infect/1000, label="Infected", color="red", linewidth=6)

ax5.set_title("Infected", fontsize = 35)

#ax5.tick_params(axis="both", labelsize=20)
#ax5.set_xlabel("Time", fontsize = 30)
#ax5.set_ylabel("Population Size (in Thousands)", fontsize = 30)

#ax5.legend(fontsize = 30, loc='center left')

#get q and suscept pop

#grid and solve non lin vars
b1Range = (0, 500) #modify to get finer results
b2Range = (0, 5)
betaVarsResol = [100, 15]

#linVarsV, nonLinVarsV = sirdv_fd.solveAllVars(susceptV, infect, recovV, dead, vacc, [b1Range, b2Range], betaVarsResol, printOut=True)
#linVars, nonLinVars = sird_fd.solveAllVars(suscept, infect, recov, dead, [b1Range, b2Range], betaVarsResol, printOut=True)


#grid and solve non lin vars

#daysToPredict = 150


#plot
#betaTime = sird_fd.getBetaTime(suscept, infect, recov, dead, linVars, nonLinVars)
#linVarsTime = sird_time.getLinVars(suscept, infect, recov, dead)
#linVarsConst = sird.getLinVars(suscept, infect, recov, dead)

betaTimeV = sirdv_fd.getBetaTime(susceptV, infect, recovV, dead, vacc, linVarsV, nonLinVarsV)
linVarsTimeV = sirdv_time.getLinVars(susceptV, infect, recovV, dead, vacc)
linVarsConstV = sirdv.getLinVars(susceptV, infect, recovV, dead, vacc)

#ax.plot(betaTime, color="green", linestyle="dashed", label="Feedback", linewidth=lW*) #feedback beta
#ax.plot(np.ones(len(linVarsTime[:,0]))*linVarsConst[0], linestyle="dashed", label="Constant", color="blue", linewidth=lW) #const

#ax.plot(linVarsTime[:,0], color="red", label="Actual", linewidth=lW/6) #time varying beta vacc

ax.plot(linVarsTimeV[:,0], color="red", label="Actual", linewidth=lW*.6) #time varying beta vacc
ax.plot(betaTimeV, color="green", linestyle="dashed", label="Feedback & Vacc.", linewidth=lW) #feedback beta vacc
ax.plot(np.ones(len(linVarsTimeV[:,0]))*linVarsConstV[0], linestyle="dotted", label="Constant & Vacc.", color="blue", linewidth=lW) #vacc

#Customizing the Figure
ax.tick_params(axis="both", labelsize=20)

ax.set_xlabel("Time", fontsize = 30)
ax.set_ylabel("Beta", fontsize = 30)

ax.legend(fontsize = 30)
ax.set_ylim([0,.25])
#plot


#plot2
dTP = 125


#linVarsConst2 = sird.getLinVars(suscept[:-dTP], infect[:-dTP], recov[:-dTP], dead[:-dTP]) #use this on initial spike

#sp, ipConst, rp, dp = sird.predictMatch(suscept, infect, recov, dead, dTP, linVars=linVarsConst, graph=False)
sp, ipConstV, rp, dp = sirdv.predictMatch(susceptV, infect, recovV, dead, vacc, dTP, linVars=linVarsConstV, graph=False)
sp, ipFeedV, rp, dp = sirdv_fd.predictMatch(susceptV, infect, recovV, dead, vacc, dTP, linVars=linVarsV, nonLinVars=nonLinVarsV, graph=False)
#sp, ipFeed, rp, dp = sird_fd.predictMatch(suscept, infect, recov, dead, dTP, linVars=linVars, nonLinVars=nonLinVars, graph=False)

ax2.plot(infect/1000, color="red", label="Actual", linewidth=lW)
#ax2.plot(ipConst/1000, color="blue", label="Constant", linestyle="dotted", linewidth=lW)
#ax2.plot(ipFeed/1000, color="green", label="Feedback", linestyle="dashed", linewidth=lW)
ax2.plot(ipConstV/1000, color="blue", label="Constant & Vacc.", linestyle="dotted", linewidth=lW)
ax2.plot(ipFeedV/1000, color="green", label="Feedback & Vacc.", linestyle="dashed", linewidth=lW)

ax2.axvline(len(suscept)-dTP, color='black', linestyle='dotted', linewidth=lW)

#Customizing the Figure
ax2.tick_params(axis="both", labelsize=20)

ax2.set_xlabel("Time", fontsize = 30)
ax2.set_ylabel("Infected (in Thousands)", fontsize = 30)

ax2.set_ylim([0, max(infect)*1.2/1000])

ax2.legend(fontsize = 30)
#plot



#feed params
#q = 0.30117300575
#Delay = 14
#FB weight decay = 0.94
#FBV weight decay = 0.94
#Solution:
#b0:  0.061108706122272954
#b1:  0.005273168933039929
#g:   0.08520091987532212
#nu:  0.0010161885990735459
#b2:  130.0
#b3:  4.999999999999999
#cost:  15179.877957803526
#linVars = [ 0.061108706122272954, 0.005273168933039929, 0.08520091987532212, 0.0010161885990735459]
#nonLinVars = [130.0, 4.999999999999999]
#q = 0.30117300575

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
#q = 0.256939564
#Delay = 21
#FB weight decay = 0.97
#Solution:
#b0:  0.03678200016916789
#b1:  0.11127071505615913
#g:   0.07836493097189762
#nu:  0.0013601273410598976
#b2:  300.0
#b3:  1.0
#cost:  579450.2696212075
#linVars = [ 0.03678200016916789, 0.11127071505615913, 0.07836493097189762, 0.0013601273410598976]
#nonLinVars = [300.0, 1.0]
#q = 0.256939564

#params for all California:
#q = 0.30117300575
#Delay = 21
#FB weight decay = 0.97
#Solution:
#b0:  -2.073135520955533
#b1:  2.1584841023683152
#g:   0.0818563787982533
#nu:  0.0013258329826889086
#b2:  10.0
#b3:  4.999999999999999
#cost:  670786.5504532194
#linVars = [ -2.073135520955533, 2.1584841023683152, 0.0818563787982533, 0.0013258329826889086]
#nonLinVars = [10.0, 4.999999999999999]
#q = 0.30117300575

def getFitError(I, IP):
    return sum((I - IP)**2) #squared error

#gridding functions
def getFitLin(suscept, infect, recov, dead, linVars, initDays):
    resol = 25 #test 25 values above 1 and 25 values below
    
    minError = 1e100 #let this be an starting enormous error
    minVal = 0
    for i in range(1, resol): #test vals above 1
        infectTest = infect.copy()
        infectTest[0:initDays] = infectTest[0:initDays] * resol / i
        
        sp,ip,rp,dp = sird.predictMatch(suscept, infectTest, recov, dead, dTP, linVars=linVars, graph=False)
        currError = getFitError(infect, ip)
        
        if(currError < minError):
            minError = currError
            minVal = resol/i
    
    for i in range(1, resol+1): #test vals below 1
        infectTest = infect.copy()
        infectTest[0:initDays] = infectTest[0:initDays] * i / resol #basically scale between 0 and 1
        
        sp,ip,rp,dp = sird.predictMatch(suscept, infectTest, recov, dead, len(suscept)-initDays, linVars=linVars, graph=False)
        currError = getFitError(infect, ip)
        
        if(currError < minError):
            minError = currError
            minVal = i/resol
        
    print("Scale by:", minVal)
    
    
    #calculate min I
    infectTest = infect.copy()
    infectTest[0:initDays] = infectTest[0:initDays] * minVal #scaler
    sp,ip,rp,dp = sird.predictMatch(suscept, infectTest, recov, dead, len(suscept)-initDays, linVars=linVars, graph=False)
    
    return ip

def getFitFeed(suscept, infect, recov, dead, linVars, nonLinVars, initDays):
    resol = 25
    
    minError = 1e100 #let this be an starting enormous error
    minVal = 0
    for i in range(1,resol+1):
        infectTest = infect.copy()
        for j in range(0, initDays):
            infectTest[j] = infectTest[j] * (i/resol)**(initDays - j) #weight decay
            
        sp,ip,rp,dp = sird_fd.predictMatch(suscept, infectTest, recov, dead, len(suscept)-initDays, linVars=linVars, nonLinVars=nonLinVars, graph=False)
        currError = getFitError(infect, ip)
            
        if(currError < minError):
            minError = currError
            minVal = i/resol
    
    print("Weight by:", minVal)
    
    #calculate min I
    infectTest = infect.copy()
    for j in range(0, initDays):
        infectTest[j] = infectTest[j] * (minVal)**(initDays - j) #weight decay
    sp,ip,rp,dp = sird_fd.predictMatch(suscept, infectTest, recov, dead, len(suscept)-initDays, linVars=linVars, nonLinVars=nonLinVars, graph=False)

    return ip
#gridding functions

#fig3, ax3 = plt.subplots(figsize=(18,8))
#ax3.set_title("Optimal Initial Conditions and Fit in Italy", fontsize = 35)

#ipFeed = getFitFeed(suscept, infect, recov, dead, linVars, nonLinVars, len(suscept)-dTP)
#ipConst = getFitLin(suscept, infect, recov, dead, linVarsConst, len(suscept)-dTP)

#ax3.plot(infect, color="red", label="Actual")
#ax3.plot(ipConst, color="blue", label="Constant", linestyle="dotted")
#ax3.plot(ipFeed, color="green", label="Feedback", linestyle="dashed")


#ax3.axvline(len(suscept)-dTP, color='black', linestyle='dashed', label="Initial Cond.")

#Customizing the Figure
#ax3.tick_params(axis="both", labelsize=20)

#ax3.set_xlabel("Time in Days", fontsize = 30)
#ax3.set_ylabel("Infected", fontsize = 30)

#ax3.legend(fontsize = 30, loc='upper right')

#ax3.set_ylim([0, max(infect)*1.2])

plt.show()

process.writeCSV(ax, dates, "..\\Figures\\CSVs\\betaCalVacc.csv")
process.writeCSV(ax2, dates, "..\\Figures\\CSVs\\predictCalVacc.csv")
process.writeCSV(ax5, dates, "..\\Figures\\CSVs\\SIRDV.csv")

import Models.SIRD as sird
import Models.SIRD_Time as sird_time
import Models.SIRD_Beta_Time as sird_beta

import Models.SAIRD as saird
import Models.SAIRD_Time as saird_time

import Models.SAIRD_Feedback as saird_fb

import Models.SIRD_Feedback as sird_fb
import Models.SIRD_Feedback_Delay as sird_fd

import Models.process as process

import numpy as np
import csv
import matplotlib.pyplot as plt
import platform



#pathc = "../Data/Italian Data/"
pathc="../Data/State Data/"
if platform.system() == "Windows":
    pathc.replace("/", "\\")
    
#filename = "National Data.csv"
filename = "CA.csv"
dates, infectRaw, recovRaw, deadRaw = process.loadData(pathc + filename)

recovRaw = process.getRecov(infectRaw, deadRaw)
infectRaw = process.getCurrentInfect(infectRaw, recovRaw, deadRaw)

#pop = 60000000
pop = 40000000



skipDays = 0
numDays = len(infectRaw) #just to get initial beginning data
#asympt = asymptRaw[skipDays:numDays]
infect = infectRaw[skipDays:numDays]
recov = recovRaw[skipDays:numDays] 
dead = deadRaw[skipDays:numDays]

daysToPredict = 150



sird.weightDecay= .98
sird.regularizer=10

sird_fd.weightDecay = .98 #very small amount of decay
sird_fd.regularizer = 10
sird_fd.betaUseDecay = True

sird_fd.delay = 21


q = sird.getQ(infect,recov, dead, pop) #use non feedback model to get q value, should be accurate enough
print("q =", q)

#q=.011
suscept = process.getSuscept(infect,recov,dead, q,pop)



b1Range = (0, 5000) #modify to get finer results
b2Range = (0, 5)
betaVarsResol = [100, 10]

linVars, nonLinVars = sird_fd.solveAllVars(suscept, infect, recov, dead, [b1Range, b2Range], betaVarsResol, printOut=True)




betaTime = sird_fd.getBetaTime(suscept, infect, recov, dead, linVars, nonLinVars)
linVarsTime = sird_time.getLinVars(suscept, infect, recov, dead)
linVarsConst = sird.getLinVars(suscept, infect, recov, dead)


fig, ax = plt.subplots(figsize=(18,8))
ax.plot(linVarsTime[:,0], color="red", label="Actual") #time varying beta
ax.plot(np.ones(len(linVarsTime[:,0]))*linVarsConst[0], color="blue", linestyle="dotted", label="Constant") #constant beta
ax.plot(betaTime, color="green", linestyle="dashed", label="Feedback") #feedback beta

#Customizing the Figure
ax.tick_params(axis="both", labelsize=20)

ax.set_title("Comparing Tracking Beta in CA (All Days)", fontsize = 35)
ax.set_xlabel("Time in Days", fontsize = 30)
ax.set_ylabel("Beta", fontsize = 30)

ax.legend(fontsize = 30)
ax.set_ylim([0,.25])

plt.show()
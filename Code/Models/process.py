import numpy as np
from scipy.stats import lognorm
import csv

#this model is for simple pre processing steps like approximating recovered and vaccinations

#scale the infections by how many tests are conducted, scaling is arbitrary so the maxes of before and after remmain the same
def scaleNewInfections(newInfect, newTests): 
    
    scaledInfections = np.zeros(len(newTests))
    
    for i in range(len(newTests)):
        if(newTests[i] != 0):
            scaledInfections[i] = newInfect[i]/newTests[i]
    
    scalingValue = 1
    
    for i in range(len(newTests)): #find the minimum scaling value, basically modI >= I for all but as a close to I as possible
        if(scaledInfections[i] != 0 and scaledInfections[i]*scalingValue < newInfect[i]):
            scalingValue = newInfect[i]/scaledInfections[i]
    
    scaledInfections = scaledInfections * scalingValue #scale somewhat appropriately
    
    return scaledInfections



def REMEDID(D): #get new infections from new deaths (not total)
    xData = np.linspace(0,100,100)
    ip = getLogNorm(xData, 5.6,5)
    iod = getLogNorm(xData, 14.5,13.2)
    dp = np.convolve(ip, iod)[:len(ip)] * (xData[1]-xData[0]) #curves * dx, dx should be one
    
    approxI = np.zeros(len(D)) #new infections
    for i in range(len(D)-len(dp)+1):
        approxI[i] = sum(D[i:i+len(dp)]*dp)
    for i in range(1,len(dp)):
        approxI[len(D)-len(dp)+i] = sum(D[len(D)-len(dp)+i:]*dp[:-i])
    return approxI

    
def getLogNorm(xData, mean, median, iters=250): #find a lognorm curve based on mean and median
    minStd = .1
    maxStd = .6

    std = 0
    bestMean = 0
    stdList = np.arange(minStd, maxStd,(minStd+maxStd)/iters)
    for currStd in stdList: #basic search
        #figure out mean based on std and median
        yData = lognorm.pdf(xData,currStd,scale=median)
        yDataMod = yData/sum(yData) #normalized
        currMean = sum(yDataMod*xData)

        if(abs(currMean-mean) < abs(bestMean-mean)):
            bestMean = currMean
            std = currStd

    yData = lognorm.pdf(xData,std,scale=median)
    yDataMod = yData/sum(yData) #normalized
    currMean = sum(yDataMod*xData)
    #print(std, currMean, mean)

    return yData


def reverseDiff(diffList):
    newList = np.zeros(len(diffList))
    newList[0] = diffList[0]
    for i in range(len(diffList)-1):
        newList[i+1] = diffList[i+1] + newList[i]
    return newList

def getRecov(totalI, D, shift=13): #approximated recoered, assume after 13 days if the new infected is not dead, they recovered
    R = np.zeros(len(totalI))
    for i in range(len(totalI) - shift):
        R[i + shift] = totalI[i] - D[i + shift]
    return R

def getCurrentInfect(totalI, R, D):
    return (totalI - R - D)

#get susceptible from q, assumes A or V are not in the model
def getSuscept(I, R, D, q, pop):
    return (q*pop) - I - R - D #S + I + R + D = q*pop

def loadData(filename): #load the dates, I, R, D data
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
    deaths = deaths.astype(float)
    recovered = recovered.astype(float)
    infected = infected.astype(float)
    return dates, infected, recovered, deaths


#for saving axes into csv files
def writeCSV(ax, dates, fileName):
    with open(fileName, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        
        headerRow = list(dates.copy())
        headerRow.insert(0, "Label")
        writer.writerow(headerRow) #top row
        
        lines = ax.get_lines() #line data
        for line in lines:
            row = list(line.get_xdata())
            row.insert(0, line.get_label())
            
            writer.writerow(row)
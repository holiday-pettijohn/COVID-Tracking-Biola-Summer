import numpy as np
import csv

#this model is for simple pre processing steps like approximating recovered and vaccinations

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
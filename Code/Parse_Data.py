# Parse_Data.py: Parses COVID data in a standardized CSV format
# JB HP

import numpy as np
import csv
import platform

class CovidTimeSeries(object):
    """ Stores the JHU time series data for a county for covid """
    def __init__(self):
        self.dates = None
        self.regionCode= None
        self.regionName=None
        self.positive=None  #Infected used by the paper
        self.Lat=None
        self.Long=None
        self.Combined_Key=None #unused
        self.healed=None   #Recovered
        self.totalCases=None
        self.tested = None
        self.deaths=None
        self.hospitalized = None
        self.vaccinated = None
        self.recentVacc = 0

class CovidTrackingDatabase(object):
    """ Stores the covid-19 data"""
    def __init__(self):
        self.CovidData={}
        self.DateRange=[]

    def loadTimeSeries(self, filenameI, startdate, enddate, fields):
        """ load the infections data from filenameI
            from startdate to enddate
        """
        csvfile=open(filenameI, newline='',  encoding='UTF-8')
        rd = csv.reader(csvfile, delimiter=',')
        data=[]
        for lv in rd:
            if(lv[0] == 'date'):
                data.insert(0,lv)
            else:
                data.insert(1,lv)
        header=data[0]

        infectionData=(data[1:])
        temp = np.array(infectionData)
        dates = temp[:,0]
        dates = dates.tolist()


        CountyD={}
        N=len(infectionData);

        for i in range(N):
            if not (dateInRange(startdate,enddate,infectionData[i][0])):
                continue
            if infectionData[i][1] not in CountyD: #if key not already initialized
                c1=CovidTimeSeries()

                fp=infectionData[i][1]
                x=fp
                c1.dates = [infectionData[i][0]]
                c1.regionName = infectionData[i][1]

                if(infectionData[i][header.index(fields["deaths"])] == ''): #Deaths
                    c1.deaths = [int(0)]
                else:
                    c1.deaths = [int(infectionData[i][header.index(fields["deaths"])])]

                if(infectionData[i][header.index(fields["infected"])] == ''): #Infected
                    c1.positive = [int(0)]
                else:
                    c1.positive = [int(infectionData[i][header.index(fields["infected"])])]

                if(infectionData[i][header.index(fields["recovered"])] == ''): #Recovered
                    c1.healed = [int(0)]
                else:
                    c1.healed = [int(infectionData[i][header.index(fields["recovered"])])]

                # optional fields which are usually not critical to models
                if("tested" in fields):
                    if(infectionData[i][header.index(fields["tested"])] == ''): #Tested
                        c1.tested = [int(0)]
                    else:
                        c1.tested = [int(infectionData[i][header.index(fields["tested"])])]

                if("hospitalized" in fields):
                    if(infectionData[i][header.index(fields["hospitalized"])] == ''): #Hospitalized
                        c1.hospitalized = [int(0)]
                    else:
                        c1.hospitalized = [int(infectionData[i][header.index(fields["hospitalized"])])]

                CountyD[x]=c1
            else: #if key already initialized
                fp=infectionData[i][1]
                x=fp
                if(infectionData[i][header.index(fields["hospitalized"])] == ''): #Hospitalized
                    CountyD[x].hospitalized.append(int(0))
                else:
                    CountyD[x].hospitalized.append(int(infectionData[i][header.index(fields["hospitalized"])]))

                if(infectionData[i][header.index(fields["infected"])] == ''): #Infected
                    CountyD[x].positive.append(int(0))
                else:
                    CountyD[x].positive.append(int(infectionData[i][header.index(fields["infected"])]))

                if(infectionData[i][header.index(fields["deaths"])] == ''): #Deaths
                    CountyD[x].deaths.append(int(0))
                else:
                    CountyD[x].deaths.append(int(infectionData[i][header.index(fields["deaths"])]))

                if(infectionData[i][header.index(fields["recovered"])] == ''): #Recovered
                    CountyD[x].healed.append(int(0))
                else:
                    CountyD[x].healed.append(int(infectionData[i][header.index(fields["recovered"])]))

                CountyD[x].dates.append(infectionData[i][0])

        for key in CountyD: #Turn the lists into arrays
            CountyD[key].deaths = np.array(CountyD[key].deaths)
            CountyD[key].positive = np.array(CountyD[key].positive)
            CountyD[key].healed = np.array(CountyD[key].healed)
            CountyD[key].totalCases = np.array(CountyD[key].totalCases)
            CountyD[key].tested = np.array(CountyD[key].tested)
            CountyD[key].dates = np.array(CountyD[key].dates)
            CountyD[key].hospitalized = np.array(CountyD[key].hospitalized)
        # assumes that no two dates are repeated and that records are consecutive
        self.DateRange=len(CountyD)
        self.CovidData=CountyD

class CovidDatabase(object):
    """ Stores the covid-19 data"""
    def __init__(self):
        self.CovidData={}
        self.DateRange=[]
      
    def loadTimeSeries(self, filenameI, filenameD, startdate, enddate):
        """ load the infections data from filenameI and death data from filenameD
            from startdate to enddate
        """
        csvfile=open(filenameI, newline='')
        rd = csv.reader(csvfile, delimiter=',')
        data=[]
        for lv in rd:
            data.append(lv)

        header=data[0]
        infectionData=data[1:]

        csvfiled=open(filenameD, newline='')
        rd = csv.reader(csvfiled, delimiter=',')
        datad=[]
        for lv in rd:
            datad.append(lv)

        headerd=datad[0]
        deathData=datad[1:]

        startdate_index=header.index(startdate) if startdate in header else header.index("1/22/20")
        enddate_index=header.index(enddate) if enddate in header else header.index("5/14/21")
        startdate_indexd=headerd.index(startdate) if startdate in headerd else headerd.index("1/22/20")
        enddate_indexd=headerd.index(enddate) if enddate in headerd else headerd.index("5/14/21")
        
        CountyD={}
        N=len(infectionData);
        for i in range(N):
            pop1=int(deathData[i][11])
            if (pop1>0):
                c1=CovidTimeSeries()
                x=infectionData[i][5]
                c1.regionName=x
                c1.dates=header[startdate_index:enddate_index+1]
                c1.positive=np.array([int(a) for a  in infectionData[i][startdate_index:enddate_index+1]])
                c1.deaths=np.array([int(a) for a  in deathData[i][startdate_indexd:enddate_indexd+1]])
                CountyD[x]=c1
        self.DateRange=header[startdate_index:enddate_index+1]
        self.CovidData=CountyD

def toWeekPeriod(filename,newName):
    csvfile=open(filename, newline='', encoding='UTF-8')
    rd = csv.reader(csvfile, delimiter=',')
    data=[]
    for lv in rd: #generating the data matrix
        data.append(lv)
    header = data[0] #get the labels
    infectionData=(data[1:]) #data without the labels
    infectionData = np.array(infectionData)
    dates = infectionData[:,0]
    N = int(np.ceil(np.shape(dates)[0] / 7))
    length = np.shape(dates)[0]
    difference = N * 7 - length
    if(difference != 0):
        N = N -1
    print(difference)

    infected = np.zeros(N * 7)
    tested  = np.zeros(N * 7)
    recovered = np.zeros(N * 7)
    deaths = np.zeros(N * 7)


    infected[:] = infectionData[:length -(7- difference),1].astype(np.float)
    tested[:] = infectionData[:length - (7- difference),2].astype(np.float)
    recovered[:] = infectionData[:length - (7- difference),3].astype(np.float)
    deaths[:] = infectionData[:length - (7- difference),4].astype(np.float)

    newDates = np.zeros((N))
    newDates = newDates.tolist()
    newInfect = np.zeros((N))
    newTested = np.zeros(N)
    newRecov = np.zeros(N)
    newDead = np.zeros(N)
    wanted = np.zeros((N,5))
    wanted = wanted.tolist()
    for i in range(N):
        temp = i * 7
        newDates[i] = dates[temp]
        newInfect[i] = infected[temp] + infected[temp + 1] + infected[temp + 2] + infected[temp + 3] + infected[temp + 4] + infected[temp + 5] + infected[temp + 6]
        newRecov[i] = recovered[temp] + recovered[temp + 1] + recovered[temp + 2] + recovered[temp + 3] + recovered[temp + 4] + recovered[temp + 5] + recovered[temp + 6]
        newDead[i] = deaths[temp] + deaths[temp + 1] + deaths[temp + 2] + deaths[temp + 3] + deaths[temp + 4] + deaths[temp + 5] + deaths[temp + 6]
        newTested[i] = tested[temp] + tested[temp + 1] + tested[temp + 2] + tested[temp + 3] + tested[temp + 4] + tested[temp + 5] + tested[temp + 6]
        wanted[i][0] = newDates[i].astype(np.str)
        wanted[i][1] = newInfect[i].astype(np.str)
        wanted[i][2] = newTested[i].astype(np.str)
        wanted[i][3] = newRecov[i].astype(np.str)
        wanted[i][4] = newDead[i].astype(np.str)
    fields = ['Dates', 'Infected', 'Tested', 'Recovered', 'Deaths']
    # writing to csv file
    with open(newName + "(Weeks).csv" , 'w', newline = '') as csvfile:
        # creating a csv writer object
        csvwriter = csv.writer(csvfile, delimiter = ',')

        # writing the fields
        csvwriter.writerow(fields)

        # writing the data rows
        csvwriter.writerows(wanted)


def addVacc(filenameV,state, startdate, enddate):
    #Load all vaccination data
    csvfile = open(filenameV, newline = '', encoding = 'UTF-8')
    rd = csv.reader(csvfile, delimiter = ',')
    dataV = []
    for lv in rd:
            dataV.append(lv)
    headerV = dataV[0]
    infectionDataV = (dataV[1:])
    temp = np.array(infectionDataV)

    #Pull date and state data
    datesV = temp[:,0]
    datesV = datesV.tolist()
    states = temp[:,1]
    states = states.tolist()

    #Get the indexes for the state wanted
    firstIndex = states.index(state)
    lastIndex = firstIndex
    while(states[lastIndex] == state):
        lastIndex += 1

    #Put state data into an array
    newData = np.zeros((lastIndex - firstIndex + 1, 2))
    newData = newData.astype(np.str)
    newData[:,0] = temp[firstIndex:lastIndex+1,0]
    newData[:,1] = temp[firstIndex:lastIndex+1,7]

    #Get correct date indexes
    temp2 = newData[:,0].tolist()
    startIndex = temp2.index(startdate)
    endIndex = temp2.index(enddate)

    #Put correct data in vacc array
    vacc = newData[startIndex:endIndex+1,1]

    #Convert all empty cells to the previous vaccination count
    preV = '0'
    for i in range(np.shape(vacc)[0]):
        if(vacc[i] == ''):
            vacc[i] = preV
        else:
            preV = vacc[i]
    vacc = vacc.astype(np.float)
    return vacc

def dateInRange(startdate, enddate, date): # See if a date is within the desired range
    #Get start date in desired format
    sDate = np.array(startdate.split('/'))
    if (len(sDate[0]) == 1):
        sDate[0] = '0' + sDate[0]
    if (len(sDate[1]) == 1):
        sDate[1] = '0' + sDate[1]
    sDateT = sDate[2] + sDate[0] + sDate[1]
    sDateT = int(sDateT)

    #Get end date in desired format
    eDate = np.array(enddate.split('/'))
    if (len(eDate[0]) == 1):
        eDate[0] = '0' + eDate[0]
    if (len(eDate[1]) == 1):
        eDate[1] = '0' + eDate[1]
    eDateT = eDate[2] + eDate[0] + eDate[1]
    eDateT = int(eDateT)

    #Get actual date in desired format
    date = np.array(date.split('/'))
    if (len(date[0]) == 1):
        date[0] = '0' + date[0]
    if (len(date[1]) == 1):
        date[1] = '0' + date[1]
    dateT = date[2] + date[0] + date[1]
    dateT = int(dateT)

    if sDateT<=dateT<=eDateT:
        return True
    else:
        return False

def writeRegionData(filename, startdate, enddate, region, fields=None):
    if fields is None:
        fields = {"deaths": "death",
                  "infected": "positive",
                  "hospitalized": "hospitalized",
                  "recovered": "recovered"}
    database=CovidTrackingDatabase();
    database.loadTimeSeries(filename, startdate, enddate, fields)
    CountyD=database.CovidData

    for key in CountyD:
        wanted = np.zeros((np.shape(CountyD[key].dates)[0],5))
        wanted = wanted.tolist()
        break
    for key in CountyD:
        if CountyD[key].regionName == region:
            tempDeaths = CountyD[key].deaths
            tempRecovered = CountyD[key].healed
            tempInfected = CountyD[key].positive
            tempHosp = CountyD[key].hospitalized
            tempDates = CountyD[key].dates
            for N in range(np.shape(CountyD[key].dates)[0]):
                wanted[N][0] = tempDates[N]
                wanted[N][1] = tempInfected[N]
                wanted[N][2] = tempHosp[N]
                wanted[N][3] = tempRecovered[N]
                wanted[N][4] = tempDeaths[N]

    filename = patho+region+".csv"
    fields = ['Dates', 'Infected','Vaccinated', 'Recovered', 'Deaths']
    # writing to csv file 
    with open(filename, 'w', newline = '') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile, delimiter = ',') 

        # writing the fields 
        csvwriter.writerow(fields) 

        # writing the data rows 
        csvwriter.writerows(wanted)

def writeCountyData(filenamei, filenamed, startdate, enddate, county):
    database=CovidDatabase();
    database.loadTimeSeries(filenamei, filenamed, startdate, enddate)
    CountyD=database.CovidData

    for key in CountyD:
        wanted = np.zeros((np.shape(CountyD[key].dates)[0],5))
        wanted = wanted.tolist()
        break
    for key in CountyD:
        if CountyD[key].regionName == region:
            tempDeaths = CountyD[key].deaths
            tempInfected = CountyD[key].positive
            tempDates = CountyD[key].dates
            for N in range(np.shape(CountyD[key].dates)[0]):
                wanted[N][0] = tempDates[N]
                wanted[N][1] = tempInfected[N]
                wanted[N][4] = tempDeaths[N]

    filename = patho+region+".csv"
    fields = ['Dates', 'Infected','Vaccinated', 'Recovered', 'Deaths']
    # writing to csv file 
    with open(filename, 'w', newline = '') as csvfile: 
        # creating a csv writer object 
        csvwriter = csv.writer(csvfile, delimiter = ',') 

        # writing the fields 
        csvwriter.writerow(fields) 

        # writing the data rows 
        csvwriter.writerows(wanted)
        
# system path to the input data
pathc = "../Data/JHU Data/"

# system path to the output data
patho = "../Data/County Data/"

if platform.system() == "Windows":
    pathc.replace("/", "\\")

# name of file to extract data from
filenamei="time_series_covid19_confirmed_US.csv"
filenamed="time_series_covid19_deaths_US.csv"

full_filenamei = pathc+filenamei
full_filenamed = pathc+filenamed

# start and end date - these do not need to be in the file
startdate="3/21/20"
enddate="5/14/21"

# region or state id
region = "Baldwin"

# fields in data file
"""
fields = {"deaths": "death",
          "infected": "positive",
          "hospitalized": "hospitalized",
          "recovered": "recovered"}
"""
writeCountyData(full_filenamei, full_filenamed, startdate, enddate, region)

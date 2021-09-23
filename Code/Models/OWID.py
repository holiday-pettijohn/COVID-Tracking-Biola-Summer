import numpy as np
import csv
import Models.process as process
import platform

#file for loading our world in data data
# https://github.com/owid/covid-19-data/blob/master/public/data/owid-covid-data.csv
# https://ourworldindata.org/

#load the data and header

fileName = "../Data/OWID Data/owid-covid-data.csv" #mac file address
#pathc="../Data/Covid Tracking State Data/"
if platform.system() == "Windows":
    fileName.replace("/", "\\")
    
csvFile = open(fileName, newline='', encoding='UTF-8')
rd = csv.reader(csvFile, delimiter=',') #reader
data=[] #the data of all countries
for lv in rd: #generating the data matrix
    data.append(lv)
    
labels = np.array(data[0]) #get the labels
data = data[1:] #remove the header
data = np.array(data) #convert to np array for simplicity

for i in range(len(data)):
    for j in range(len(data[i])):
        if(data[i,j] == ''): #empty cell, convert to 0
            data[i,j] = '0'

countryColumn = np.where(labels == "location")[0][0] #the country column
datesColunn = np.where(labels == "date")[0][0] #the dates column

def printLabels(): #for convenience of figuring out which columns you need
    for i in range(len(labels)):
        print(i, "\t" + labels[i])

#select columns are for which data columns you want (dates will be returned always, don't select columns 1-3
def LoadCountry(countryName, selectColumns=np.arange(len(data[0]))): #load the data from a particular country
    countryData = data[ data[:,countryColumn] == countryName] #all the selected columns from that country
    dates = countryData[:,datesColunn]
    
    return dates, countryData[:, selectColumns].astype(np.float).transpose()



def LoadCountryNormal(countryName): #load the country data with typical processing
    dates, countryData = LoadCountry(countryName, selectColumns=[4, 5, 7, 25, 35, 46])
    [totalI, newI, D, tests, V, pop] = countryData
    
    pop = pop[0] #get as a number, instead of a list
    
    #change values so that S+I+R+D+V = 1
    totalI = totalI/pop
    D = D/pop
    V = V/pop
    totalI = totalI/pop
    
    modNewI = process.scaleNewInfections(newI, tests) #scale by tests
    
    I = process.reverseDiff(modNewI) #aggregate
    I = I * (.25/(max(I))) #adjust so it matches max = .1, this is arbitrary
    
    R = process.getRecov(I, D)
    I = I - R - D #changge to current infections, instead of total
    
    fNZ = 0 #first date with nonzero data
    lNZ = len(I)-1 #last date with nonzero data
    
    for i in range(len(I)):
        if(I[i] != 0):
            fNZ = i
            break
         
    for i in range(len(I)):
        if(I[-i] != 0):
            lNZ = len(I) - i - 1
            break
    
    dates = dates[fNZ:lNZ]
    I = I[fNZ:lNZ]
    R = R[fNZ:lNZ]
    D = D[fNZ:lNZ]
    V = V[fNZ:lNZ]
    
    sD = 0 #startDate, first day with over .001 infection
    while(I[sD] < .001):
        sD = sD+1
    
    return dates[sD:],I[sD:],R[sD:],D[sD:],V[sD:] #current infections, recoveries, deaths, vaccinations
    
def LoadCountryNormalDeaths(countryName, shiftAmount=15): #load the country data with typical processing via deaths
    dates, countryData = LoadCountry(countryName, selectColumns=[4, 7, 36, 46])
    [totalI, D, V, pop] = countryData
    
    pop = pop[0] #get as a number, instead of a list
    #change values so that S+I+R+D+V = 1
    D = D/pop
    V = V/pop
    totalI = totalI/pop
    
    modI = D[shiftAmount:] #infection is just a scaled version of deaths with some shift
    
    modI = modI*(.25/(max(modI))) #adjust so it matches max = .25, this is arbitrary #change modI so the totals are the same
    
    dates = dates[:-shiftAmount]
    V = V[:-shiftAmount]
    D = D[:-shiftAmount]
    
    R = process.getRecov(modI, D, shift=13)
    I = modI - R - D #changge to current infections, instead of total
    
    
    sD = 0 #startDate, first day with over .001 infection
    while(I[sD] < .001):
        sD = sD+1
    
    return dates[sD:],I[sD:],R[sD:],D[sD:],V[sD:] #current infections, recoveries, deaths, vaccinations
                                 
                                     
# 0 	iso_code
# 1 	continent
# 2 	location
# 3 	date
# 4 	total_cases
# 5 	new_cases
# 6 	new_cases_smoothed
# 7 	total_deaths
# 8 	new_deaths
# 9 	new_deaths_smoothed
# 10 	total_cases_per_million
# 11 	new_cases_per_million
# 12 	new_cases_smoothed_per_million
# 13 	total_deaths_per_million
# 14 	new_deaths_per_million
# 15 	new_deaths_smoothed_per_million
# 16 	reproduction_rate
# 17 	icu_patients
# 18 	icu_patients_per_million
# 19 	hosp_patients
# 20 	hosp_patients_per_million
# 21 	weekly_icu_admissions
# 22 	weekly_icu_admissions_per_million
# 23 	weekly_hosp_admissions
# 24 	weekly_hosp_admissions_per_million
# 25 	new_tests
# 26 	total_tests
# 27 	total_tests_per_thousand
# 28 	new_tests_per_thousand
# 29 	new_tests_smoothed
# 30 	new_tests_smoothed_per_thousand
# 31 	positive_rate
# 32 	tests_per_case
# 33 	tests_units
# 34 	total_vaccinations
# 35 	people_vaccinated
# 36 	people_fully_vaccinated
# 37 	total_boosters
# 38 	new_vaccinations
# 39 	new_vaccinations_smoothed
# 40 	total_vaccinations_per_hundred
# 41 	people_vaccinated_per_hundred
# 42 	people_fully_vaccinated_per_hundred
# 43 	total_boosters_per_hundred
# 44 	new_vaccinations_smoothed_per_million
# 45 	stringency_index
# 46 	population
# 47 	population_density
# 48 	median_age
# 49 	aged_65_older
# 50 	aged_70_older
# 51 	gdp_per_capita
# 52 	extreme_poverty
# 53 	cardiovasc_death_rate
# 54 	diabetes_prevalence
# 55 	female_smokers
# 56 	male_smokers
# 57 	handwashing_facilities
# 58 	hospital_beds_per_thousand
# 59 	life_expectancy
# 60 	human_development_index
# 61 	excess_mortality
import requests
import csv
import urllib.request
from bs4 import BeautifulSoup
from urllib.request import urlopen
import re
from datetime import datetime

def cell2CumulNewCases(inStr):
    criterion = re.compile("^[0-9]+$")
    if inStr == "":
        cumulCases, newCases = 0, 0
    elif type(inStr) == int:
        cumulCases, newCases = inStr, 0
    elif criterion.fullmatch(inStr):
        cumulCases, newCases = int(process_num(inStr)), 0
    else:
        criterion = re.compile("^[0-9]+\([^\)][0-9]+\)")
        idxBra = inStr.find('(')
        idxKet = inStr.find(')')
        cumulCases = int(process_num(inStr[0:idxBra]))
        newCases = int(process_num(inStr[(idxBra+1):idxKet]))

    return cumulCases, newCases

def cumulNewCases2Cell(cumulNewCases):
    return cumulAndNewCases2Cell(cumulNewCases[0], cumulNewCases[1])

def cumulAndNewCases2Cell(cumulCases, newCases):
    if newCases == 0:
        return cumulCases
    else:
        return str(cumulCases) + "(" + str(newCases) + ")"

def process_num(num):
    return float(re.sub(r'[^\w\s.]','',num))

def getWebTableByCaption2CSV(url, captionWanted, filename):
    html = urlopen(url)

    soup = BeautifulSoup(html, 'html.parser')
    for caption in soup.find_all('caption'):
        if captionWanted in caption.get_text():
            table = caption.find_parent('table')
            print("found data by states")
    
    output_headers = []
    output_rows = []
    for table_row in table.findAll('tr'):
        headers = table_row.findAll('th')
        output_header = []
        for header in headers:
            output_header.append(str.rstrip(header.text))
        if(output_header != []):
           output_headers.append(output_header)
    
        columns = table_row.findAll('td')
        output_row = []
        for column in columns:
            cell = str.rstrip(column.text)
            output_row.append(cell)
        if(output_row != []):
            output_rows.append(output_row)
    
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(output_headers[0])
        writer.writerows(output_rows)

url = "https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Malaysia#Cases_by_state"
# find a table with "Distribution of cumulative confirmed cases in various administrative regions of Malaysia"
captionWanted = "Distribution of cumulative confirmed cases in various administrative regions of Malaysia"
rawCSVFilename = '../../data/byStatesRaw.csv'
getWebTableByCaption2CSV(url, captionWanted, rawCSVFilename)

with open(rawCSVFilename, newline='') as csvfile:
    data = list(csv.reader(csvfile))
# below is specific to the table on Malaysia's Wikipedia page

# remove line with date "4/3-12/3" and "15/3"
data = [row for row in data if '4/3-12/3' not in row]
data = [row for row in data if '15/3' not in row]

# For 16/3, 17/3, 18/3, split KL+PT proportionally
for i in range(len(data)):
    if(data[i][0] == '14/3'):
        idx14_3 = i
    if(data[i][0] == '16/3'):
        idx16_3 = i
    if(data[i][0] == '17/3'):
        idx17_3 = i
    if(data[i][0] == '18/3'):
        idx18_3 = i
    if(data[i][0] == '19/3'):
        idx19_3 = i
idxKL = data[0].index('KL')
dataKL14_3 = cell2CumulNewCases(data[idx14_3][idxKL])
dataPT14_3 = cell2CumulNewCases(data[idx14_3][idxKL+1])
data16_3 = cell2CumulNewCases(data[idx16_3][idxKL])
data17_3 = cell2CumulNewCases(data[idx17_3][idxKL])
data18_3 = cell2CumulNewCases(data[idx18_3][idxKL])
dataKL19_3 = cell2CumulNewCases(data[idx19_3][idxKL])
dataPT19_3 = cell2CumulNewCases(data[idx19_3][idxKL+1])
ratioKL14_3 = dataKL14_3[0]/(dataKL14_3[0]+dataPT14_3[0])
ratioKL19_3 = dataKL19_3[0]/(dataKL19_3[0]+dataPT19_3[0])

ratioKL16_3 = ((19-16)*ratioKL14_3 + (16-14)*ratioKL19_3)/(19-14)
ratioKL17_3 = ((19-17)*ratioKL14_3 + (17-14)*ratioKL19_3)/(19-14)
ratioKL18_3 = ((19-18)*ratioKL14_3 + (18-14)*ratioKL19_3)/(19-14)

dataKL16_3 = (int(ratioKL16_3*data16_3[0]), int(ratioKL16_3*data16_3[1]))
dataKL17_3 = (int(ratioKL17_3*data17_3[0]), int(ratioKL17_3*data17_3[1]))
dataKL18_3 = (int(ratioKL18_3*data18_3[0]), int(ratioKL18_3*data18_3[1]))

ratioPT16_3 = 1 - ratioKL16_3
ratioPT17_3 = 1 - ratioKL17_3
ratioPT18_3 = 1 - ratioKL18_3

dataPT16_3 = (int(ratioPT16_3*data16_3[0]), int(ratioPT16_3*data16_3[1]))
dataPT17_3 = (int(ratioPT17_3*data17_3[0]), int(ratioPT17_3*data17_3[1]))
dataPT18_3 = (int(ratioPT18_3*data18_3[0]), int(ratioPT18_3*data18_3[1]))

data[idx16_3][idxKL] = cumulNewCases2Cell(dataKL16_3)
data[idx16_3].insert(idxKL+1, cumulNewCases2Cell(dataPT16_3)) 
data[idx17_3][idxKL] = cumulNewCases2Cell(dataKL17_3)
data[idx17_3].insert(idxKL+1, cumulNewCases2Cell(dataPT17_3)) 
data[idx18_3][idxKL] = cumulNewCases2Cell(dataKL18_3)
data[idx18_3].insert(idxKL+1, cumulNewCases2Cell(dataPT18_3)) 

# Remove the final "Source" column
for singleData in data:
    del singleData[len(singleData)-1]

def processColumnNames(columnRaw):
    switcher={
        "Date":"Date",
        "JH":"Johor",
        "KD":"Kedah",
        "KE":"Kelantan",
        "ML":"Malacca",
        "NS":"Negeri Sembilan",
        "PH":"Pahang",
        "PG":"Penang",
        "PK":"Perak",
        "PR":"Perlis",
        "SB":"Sabah",
        "SR":"Sarawak",
        "SE":"Selangor",
        "TR":"Terengganu",
        "KL":"Kuala Lumpur",
        "PT":"Putrajaya",
        "LB":"Labuan"
    }
    return switcher.get(columnRaw, "Invalid input column")

# Store variables: dates, cumulCases
columnsRaw = data[0]
columns = [""] * len(columnsRaw)
for i in range(len(columnsRaw)):
    columns[i] = processColumnNames(columnsRaw[i])
datesStr = [""] * (len(data)-1)
cumulCases = [[0 for x in range(len(columns)-1)] for y in range(len(data)-1)]
for i in range(1, len(data)):
    dateStr = data[i][0]
    day, month = dateStr.split("/")
    datesStr[i-1] = "2020-" + str(month).zfill(2) + "-" + str(day).zfill(2)

    for j in range(1, len(columns)):
        cumulCase, newCase = cell2CumulNewCases(data[i][j])
        cumulCases[i-1][j-1] = cumulCase
# save cumulative infecteds CSV
with open('../../data/totalInfectedByStates.csv', 'w') as csvfile:
    for j in range(len(columns)-1):
        csvfile.write(str(columns[j]) + ',')
    csvfile.write(str(columns[len(columns)-1]) + '\n')
    for i in range(len(cumulCases)):
        csvfile.write(datesStr[i] + ',')
        for j in range(len(cumulCases[0])-1):
            csvfile.write(str(cumulCases[i][j]) + ',')
        csvfile.write(str(cumulCases[i][len(cumulCases[0])-1]) + '\n')
# save daily new infecteds CSV
with open('../../data/newDailyInfectedByStates.csv', 'w') as csvfile:
    newDailyCases = [[0 for x in range(len(cumulCases[0]))] for y in range(len(cumulCases))]
    newDailyCases[0] = cumulCases[0]
    prevDate = datetime.strptime(datesStr[0], "%Y-%m-%d")
    for i in range(1, len(datesStr)):
        nowDate = datetime.strptime(datesStr[i], "%Y-%m-%d")
        days = (nowDate - prevDate).days
        for j in range(len(newDailyCases[0])):
            newDailyCases[i][j] = (cumulCases[i][j] - cumulCases[i-1][j]) / days
        prevDate = nowDate

    for j in range(len(columns)-1):
        csvfile.write(str(columns[j]) + ',')
    csvfile.write(str(columns[len(columns)-1]) + '\n')
    for i in range(len(newDailyCases)):
        csvfile.write(datesStr[i] + ',')
        for j in range(len(newDailyCases[0])-1):
            csvfile.write(str(newDailyCases[i][j]) + ',')
        csvfile.write(str(newDailyCases[i][len(newDailyCases[0])-1]) + '\n')

print("a")
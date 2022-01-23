import re
import csv
from datetime import datetime
import numpy as np
from numpy import savetxt

def cell2TwoNumbersByBracket(inStr):
    withoutComma = re.compile("^[0-9]+$")
    withComma = re.compile("^\d{1,3}(,\d{3})*(\.\d+)?$")
    if inStr == "":
        firstNumber, secondNumber = 0, 0
    elif type(inStr) == int:
        firstNumber, secondNumber = inStr, 0
    elif withoutComma.fullmatch(inStr):
        firstNumber, secondNumber = int(process_num(inStr)), 0
    elif withComma.fullmatch(inStr):
        firstNumber, secondNumber = int(process_num(str(inStr).replace(",",""))), 0
    else:
        withNewCaseInBracket = re.compile("^[0-9]+\([^\)][0-9]+\)")
        idxBra = inStr.find('(')
        idxKet = inStr.find(')')
        firstNumber = int(process_num(inStr[0:idxBra]))
        secondNumber = int(process_num(inStr[(idxBra+1):idxKet]))

    return firstNumber, secondNumber

def cumulNewCases2Cell(cumulNewCases):
    return cumulAndNewCases2Cell(cumulNewCases[0], cumulNewCases[1])

def cumulAndNewCases2Cell(cumulCases, newCases):
    if newCases == 0:
        return cumulCases
    else:
        return str(cumulCases) + "(" + str(newCases) + ")"

def process_num(num):
    return float(re.sub(r'[^\w\s.]','',num))

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

def createByStateData(rawCSVFilename):
    with open(rawCSVFilename, newline='', encoding='utf-8') as csvfile:
        data = list(csv.reader(csvfile))
    # below is specific to the table on Malaysia's Wikipedia page
    
    # remove line with date "4/3/2020-12/3/2020" and "15/3/2020"
    data = [row for row in data if '4/3/2020-12/3/2020' not in row]
    data = [row for row in data if '15/3/2020' not in row]
    data = [row for row in data if '15/03/2020' not in row]
    
    # For 16/3/2020, 17/3/2020, 18/3/2020, split KL+PT proportionally
    for i in range(len(data)):
        if(data[i][0] == '14/3/2020' or data[i][0] == '14/03/2020'):
            idx14_3 = i
        if(data[i][0] == '16/3/2020' or data[i][0] == '16/03/2020'):
            idx16_3 = i
        if(data[i][0] == '17/3/2020' or data[i][0] == '17/03/2020'):
            idx17_3 = i
        if(data[i][0] == '18/3/2020' or data[i][0] == '18/03/2020'):
            idx18_3 = i
        if(data[i][0] == '19/3/2020' or data[i][0] == '19/03/2020'):
            idx19_3 = i
    idxKL = data[0].index('KL')
    dataKL14_3 = cell2TwoNumbersByBracket(data[idx14_3][idxKL])
    dataPT14_3 = cell2TwoNumbersByBracket(data[idx14_3][idxKL+1])
    data16_3 = cell2TwoNumbersByBracket(data[idx16_3][idxKL])
    data17_3 = cell2TwoNumbersByBracket(data[idx17_3][idxKL])
    data18_3 = cell2TwoNumbersByBracket(data[idx18_3][idxKL])
    dataKL19_3 = cell2TwoNumbersByBracket(data[idx19_3][idxKL])
    dataPT19_3 = cell2TwoNumbersByBracket(data[idx19_3][idxKL+1])
    ratioKL14_3 = dataKL14_3[0]/(dataKL14_3[0]+dataPT14_3[0])
    ratioKL19_3 = dataKL19_3[0]/(dataKL19_3[0]+dataPT19_3[0])
    
    ratioKL16_3 = ((19-16)*ratioKL14_3 + (16-14)*ratioKL19_3)/(19-14)
    ratioKL17_3 = ((19-17)*ratioKL14_3 + (17-14)*ratioKL19_3)/(19-14)
    ratioKL18_3 = ((19-18)*ratioKL14_3 + (18-14)*ratioKL19_3)/(19-14)
    
    dataKL16_3 = (int(ratioKL16_3*data16_3[0]), int (ratioKL16_3*data16_3[1]))
    dataKL17_3 = (int(ratioKL17_3*data17_3[0]), int (ratioKL17_3*data17_3[1]))
    dataKL18_3 = (int(ratioKL18_3*data18_3[0]), int (ratioKL18_3*data18_3[1]))
    
    ratioPT16_3 = 1 - ratioKL16_3
    ratioPT17_3 = 1 - ratioKL17_3
    ratioPT18_3 = 1 - ratioKL18_3
    
    dataPT16_3 = (int(ratioPT16_3*data16_3[0]), int (ratioPT16_3*data16_3[1]))
    dataPT17_3 = (int(ratioPT17_3*data17_3[0]), int (ratioPT17_3*data17_3[1]))
    dataPT18_3 = (int(ratioPT18_3*data18_3[0]), int (ratioPT18_3*data18_3[1]))
    
    data[idx16_3][idxKL] = cumulNewCases2Cell(dataKL16_3)
    data[idx16_3].insert(idxKL+1, cumulNewCases2Cell(dataPT16_3)) 
    data[idx17_3][idxKL] = cumulNewCases2Cell(dataKL17_3)
    data[idx17_3].insert(idxKL+1, cumulNewCases2Cell(dataPT17_3)) 
    data[idx18_3][idxKL] = cumulNewCases2Cell(dataKL18_3)
    data[idx18_3].insert(idxKL+1, cumulNewCases2Cell(dataPT18_3)) 
    
    # Remove the final "Source" column
    for singleData in data:
        del singleData[len(singleData)-1]
    

    # Store variables: dates, cumulCases
    columnsRaw = data[0]
    columns = [""] * len(columnsRaw)
    for i in range(len(columnsRaw)):
        columns[i] = processColumnNames(columnsRaw[i])
    datesStr = [""] * (len(data)-1)
    cumulCases = [[0 for x in range(len(columns)-1)] for y in range (len(data)-1)]
    for i in range(1, len(data)):
        dateStr = data[i][0]
        #print(dateStr)
        #print(dateStr.split("/"))
        day, month, year = dateStr.split("/")
        datesStr[i-1] = str(year) + "-" + str(month).zfill(2) + "-" + str   (day).zfill(2)
    
        for j in range(1, len(columns)):
            cumulCase, newCase = cell2TwoNumbersByBracket(data[i][j])
            cumulCases[i-1][j-1] = cumulCase
    # save cumulative infecteds CSV
    with open('./data/totalInfectedByStates.csv', 'w') as csvfile:
        for j in range(len(columns)-1):
            csvfile.write(str(columns[j]) + ',')
        csvfile.write(str(columns[len(columns)-1]) + '\n')
        for i in range(len(cumulCases)):
            csvfile.write(datesStr[i] + ',')
            for j in range(len(cumulCases[0])-1):
                csvfile.write(str(cumulCases[i][j]) + ',')
            csvfile.write(str(cumulCases[i][len(cumulCases[0])-1]) +    '\n')
    # save daily new infecteds CSV
    with open('./data/newDailyInfectedByStates.csv', 'w') as csvfile:
        newDailyCases = [[0 for x in range(len(cumulCases[0]))] for y   in range(len(cumulCases))]
        newDailyCases[0] = cumulCases[0]
        prevDate = datetime.strptime(datesStr[0], "%Y-%m-%d")
        for i in range(1, len(datesStr)):
            nowDate = datetime.strptime(datesStr[i], "%Y-%m-%d")
            days = (nowDate - prevDate).days
            for j in range(len(newDailyCases[0])):
                newDailyCases[i][j] = (cumulCases[i][j] - cumulCases    [i-1][j]) / days
            prevDate = nowDate
    
        for j in range(len(columns)-1):
            csvfile.write(str(columns[j]) + ',')
        csvfile.write(str(columns[len(columns)-1]) + '\n')
        for i in range(len(newDailyCases)):
            csvfile.write(datesStr[i] + ',')
            for j in range(len(newDailyCases[0])-1):
                csvfile.write(str(newDailyCases[i][j]) + ',')
            csvfile.write(str(newDailyCases[i][len(newDailyCases    [0])-1]) + '\n')
    
    # dynamics talbe by states (x=cumulative cases, y1, y2, .... = new   cases per day for each states, z = date just for tooltip)
    dynamicsTableHeader = []
    dynamicsTableHeader.append("Total cases")
    for j in range(1, len(columns)):
        dynamicsTableHeader.append(columns[j])
    dynamicsTableHeader.append(columns[0])
    dynamicTableNp = []
    cumulCasesInt = []
    for i in range(len(cumulCases)):
        for j in range(1, len(columns)):
            dynamicsTableRow = []
            cumulCase = cumulCases[i][j-1] #cumulCase axis is always    integer, so no problem in vstack
            if cumulCase > 0:
                newCase = str(newDailyCases[i][j-1])
                dynamicsTableRow.append(cumulCase)
                dynamicsTableRowData = [''] * len(cumulCases[i]) # new   cases is None or integer
                dynamicsTableRow.extend(dynamicsTableRowData)
                dynamicsTableRow[j] = newCase
                dynamicsTableRow.append(datesStr[i])
                if len(dynamicTableNp) == 0:
                    dynamicTableNp = np.array(dynamicsTableRow)
                    cumulCasesInt = [cumulCase]
                else:
                    dynamicTableNp = np.vstack((dynamicTableNp,     dynamicsTableRow))
                    cumulCasesInt.append(cumulCase)
    dynamicTableNp = dynamicTableNp[np.argsort(cumulCasesInt)]
    dynamicTableNp = np.vstack((dynamicsTableHeader, dynamicTableNp))
    savetxt('./data/byStateDynamics.csv', dynamicTableNp, delimiter =   ',', fmt = '%s')
    
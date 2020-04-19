import requests
import urllib.request
import json
import numpy as np
from numpy import savetxt
from datetime import datetime

def parseFlourishTable(inputStr):
	jsonRaw = json.loads(inputStr)['rows']
	nRows = len(jsonRaw)
	matrix = []
	if(nRows == 1):
		matrix.append(jsonRaw['columns'])
	else:
		for i in range(nRows):
			matrix.append(jsonRaw[i]['columns'])
	return matrix

# start here to fetch data from a website
url = "https://public.flourish.studio/visualisation/1641110/embed"

# fetch table data
loadColumnNames = False
loadBody = False
for lineBytes in urllib.request.urlopen(url):
	line = lineBytes.decode("utf-8")
	if "_Flourish_data_column_names = " in line and loadColumnNames == False:
		columnNames = line[line.find('{"rows":'):(line.find(']}}')+3)]
		loadColumnNames = True
	if "_Flourish_data = " in line and loadBody == False:
		body = line[line.find('{"rows":'):(line.find(']}]}')+4)]
		loadBody = True
	if loadColumnNames and loadBody: break

tableHeader = parseFlourishTable(columnNames)
tableHeaderNp = np.array(tableHeader)
tableBody = parseFlourishTable(body)
# convert date DD/MMM to DD/MM/YYYY format
for i in range(len(tableBody)):
	dateRaw = tableBody[i][0]
	day, monthStr = dateRaw.split("-")
	month = datetime.strptime(monthStr, '%b').month
	dateStr = "2020-" + str(month).zfill(2) + "-" + str(day)
	tableBody[i][0]  = dateStr
tableBodyNp = np.flip(np.array(tableBody), axis = 0)
table = np.append(tableHeaderNp, tableBodyNp, axis = 0)

# save a total number file
savetxt('../../data/totalRaw.csv', table, delimiter = ',', fmt = '%s')

# Date, total cases, new cases, total death, total recovered --> Date, total cases, total death, total recovered, active cases
table[0][2] = table[0][3]
table[0][3] = table[0][4]
table[0][4] = 'Active cases'

for i in range(1, len(table)):
    totalDeath = table[i][3]
    totalRecovered = table[i][4]
    activeCases = str(int(table[i][1]) - int(table[i][3]) - int(table[i][4]))
    table[i][2] = totalDeath
    table[i][3] = totalRecovered
    table[i][4] = activeCases

# save a total number file with active cases instead of new cases
savetxt('../../data/dailyTotal.csv', table, delimiter = ',', fmt = '%s')

# dynamics table (x=cumulative cases, y = new cases per day, z = date just for tooltip)
dynamicsTableOriginal = np.append(tableHeaderNp, tableBodyNp, axis = 0)
dynamicsTable = []
dynamicsTableHeader = ["Total cases", "New cases per day", "Date"]
dynamicsTable.append(dynamicsTableHeader)
dynamicsTableRow = [table[1][1], table[1][1], table[1][0]]
dynamicsTable.append(dynamicsTableRow)
for i in range(2, len(table)):
    dynamicsTableRow = [table[i][1], int(table[i][1]) - int(table[i-1][1]), table[i][0]]
    dynamicsTable.append(dynamicsTableRow)

savetxt('../../data/dailyTotalDynamics.csv', dynamicsTable, delimiter = ',', fmt = '%s')

print("Finished")


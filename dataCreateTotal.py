from numpy import savetxt
import numpy as np

def createTotalData(table):
    # Date, total cases, new cases, total death, total recovered --> Date, total cases, total death, total   recovered, active cases
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
    savetxt('./data/dailyTotal.csv', table, delimiter = ',', fmt = '%s')
    
    # the latest date, total death, total recovered, active cases
    lastIdx = len(table) - 1
    latestTotalTable =np.array([['name', 'value'],
                                [table[0][0], table[lastIdx][0]], # date
                                [table[0][1], table[lastIdx][1]], # total
                                [table[0][2], table[lastIdx][2]], # death
                                [table[0][3], table[lastIdx][3]], # recovered
                                [table[0][4], table[lastIdx][4]]]) # active
    savetxt('./data/latestTotal.csv', latestTotalTable, delimiter = ',', fmt = '%s')
    
    # Date, total change of cases, death, recovered, active cases
    tableChange =np.copy(table)
    for i in range(2, len(table)):
        for j in range(1, len(table[0])):
            tableChange[i][j] = str(int(table[i][j]) - int(table[i-1][j]))
    savetxt('./data/dailyTotalChange.csv', tableChange, delimiter = ',', fmt = '%s')
    
    # dynamics table (x=cumulative cases, y = new cases per day, z = date just for tooltip)
    dynamicsTable = []
    dynamicsTableHeader = ["Total cases", "New cases per day", "Date"]
    dynamicsTable.append(dynamicsTableHeader)
    dynamicsTableRow = [table[1][1], table[1][1], table[1][0]]
    dynamicsTable.append(dynamicsTableRow)
    for i in range(2, len(table)):
        dynamicsTableRow = [table[i][1], int(table[i][1]) - int(table[i-1][1]), table[i][0]]
        dynamicsTable.append(dynamicsTableRow)
    
    savetxt('./data/dailyTotalDynamics.csv', dynamicsTable, delimiter = ',', fmt = '%s')


import pandas as pd
import numpy as np
from datetime import timedelta, datetime

caseFileByState = "../data/totalInfectedByStates.csv"
caseByState = pd.read_csv(caseFileByState)
caseFileNationwide = "../data/dailyTotal.csv"
caseNationwides = pd.read_csv(caseFileNationwide)

columnNames = ['state', 'date', 'cases']
df_cases = pd.DataFrame(columns = columnNames)

# convert "date, state1, state2, ..." to "state(name), date, cases"
nByState = len(caseByState)
nStates = len(caseByState.columns)
nNationwide = len(caseNationwides)

datesByState = caseByState['Date']
i_df = 0
for j in range(1, nStates):
    stateName = caseByState.columns[j]
    stateCases = caseByState[stateName]
    iFirstCase = stateCases.ne(0).idxmax()

    for i in range(iFirstCase, nByState):
        if i == 0 or (datetime.strptime(datesByState[i], "%Y-%m-%d") - datetime.strptime(datesByState[i-1], "%Y-%m-%d")).days == 1:
            df_cases.loc[i_df] = [stateName, datesByState[i], stateCases[i]]
            i_df += 1
        else:
            nextDay = datetime.strptime(datesByState[i], "%Y-%m-%d")
            prevDay = datetime.strptime(datesByState[i-1], "%Y-%m-%d")
            days = (nextDay - prevDay).days
            nextCases = float(stateCases[i])
            prevCases = float(stateCases[i-1])
            slope = (nextCases - prevCases) / float(days)
            for d in range(1, days+1): # from 1 to days
                dateStr = (prevDay + timedelta(days=d)).strftime("%Y-%m-%d")
                casesStr = str(int(prevCases + slope * d))
                df_cases.loc[i_df] = [stateName, dateStr, casesStr]
                i_df += 1

        #df_cases.append([stateName, datesByState[i], stateCases[i]])

datesNationwide = caseNationwides['Date']
nationwideCases = caseNationwides['Total cases']
iFirstCase = nationwideCases.ne(0).idxmax()
for i in range(iFirstCase, nNationwide):
    df_cases.loc[i_df] = ['Malaysia', datesNationwide[i], nationwideCases[i]]
    i_df += 1

df_cases.to_csv("../rt/data/casesForRt.csv", index=False)

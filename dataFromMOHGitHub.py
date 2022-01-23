import pandas as pd
from datetime import datetime
from datetime import timedelta
import numpy as np
from dataCreateByState import cell2TwoNumbersByBracket
from dataCreateByState import cumulAndNewCases2Cell
from dataCreateByState import createByStateData
from dataCreateTotal import createTotalData
import csv

file_cases_nation = "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_malaysia.csv"
file_deaths_nation = "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_malaysia.csv"
file_case_states = "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/cases_state.csv"
#file_deaths_states = "https://raw.githubusercontent.com/MoH-Malaysia/covid19-public/main/epidemic/deaths_state.csv"

csv_cases_nation = pd.read_csv(file_cases_nation, parse_dates=["date"])
csv_deaths_nation = pd.read_csv(file_deaths_nation, parse_dates=["date"])
csv_case_states = pd.read_csv(file_case_states, parse_dates=["date"])

# load nationwide data
rawTotalCSVFilename = './data/totalRaw.csv'
file_total_raw = pd.read_csv(rawTotalCSVFilename, parse_dates=["Date"])

lastDate = file_total_raw['Date'].values[-1]
startDate = lastDate + np.timedelta64(1, "D")
endDate = csv_cases_nation['date'].values[-1]
endDate_death = csv_deaths_nation['date'].values[-1]
if(endDate != endDate_death):
    raise ValueError("Latest date from 1) national cases (" + np.datetime_as_string(endDate, "%Y-%m-%d") + ") and 2) death cases (" + np.datetime_as_string(endDate_death, "%Y-%m-%d") + ") are different");
lastRow = file_total_raw.iloc[-1]
lastTotalCases = (int)(lastRow['Total cases'])
lastNewCases = (int)(lastRow['New cases'])
lastTotalDeaths = (int)(lastRow['Total deaths'])
lastTotalRecovered = (int)(lastRow['Total recovered'])

file_total_raw_append = pd.DataFrame(columns=file_total_raw.columns)
csv_cases_nation_new = csv_cases_nation[csv_cases_nation['date'] >= startDate]
csv_deaths_nation_new = csv_deaths_nation[csv_deaths_nation['date'] >= startDate]

for i in range((int)((endDate - startDate) / np.timedelta64(1, "D")) + 1):
    date = startDate + np.timedelta64(i, "D")
    newCases = (int)(csv_cases_nation_new.iloc[i]['cases_new'])
    newRecovered = (int)(csv_cases_nation_new.iloc[i]['cases_recovered'])
    newDeaths = (int)(csv_deaths_nation_new.iloc[i]['deaths_new'])
    totalCases = newCases + lastTotalCases
    totalDeaths = newDeaths + lastTotalDeaths
    totalRecovered = newRecovered + lastTotalRecovered
    file_total_raw_append.loc[i] = {"Date": date, "Total cases": totalCases, "New cases": newCases, "Total deaths": totalDeaths, "Total recovered": totalRecovered}
    lastTotalCases = totalCases
    lastTotalDeaths = totalDeaths
    lastTotalRecovered = totalRecovered
file_total_raw_append.to_csv(rawTotalCSVFilename, mode = "a", header = False, index = False)
reader = csv.reader(open(rawTotalCSVFilename, "r"), delimiter=",")
totalTable = list(reader)
createTotalData(totalTable)

# by state data
rawByStateCSVFilename = './data/byStatesRaw.csv'
outputStateColumn = ["JH","KD","KE","ML","NS","PH","PG","PK","PR","SB","SR","SE","TR","KL","PT","LB"]
inputStateLabel = ["Johor", "Kedah", "Kelantan", "Melaka", "Negeri Sembilan", "Pahang",
                   "Pulau Pinang", "Perak", "Perlis", "Sabah", "Sarawak", "Selangor", 
                   "Terengganu", "W.P. Kuala Lumpur", "W.P. Putrajaya", "W.P. Labuan"]
# bacause byStatesRaw contains irregular date input in an old entries like "4/3/2020-12/3/2020"
# we extract the last row and convert date string to date pbject
file_by_state_raw_last_row = pd.read_csv(rawByStateCSVFilename).iloc[-1]
lastDate =  pd.to_datetime(file_by_state_raw_last_row['Date'], format="%d/%m/%Y")
startDate = lastDate + np.timedelta64(1, "D")
endDate = csv_case_states['date'].values[-1]
# format of [ cumul cases(new cases) ]
lastCases = [cell2TwoNumbersByBracket(file_by_state_raw_last_row[state])[0] for state in outputStateColumn]

file_by_state_raw_append = pd.DataFrame(columns=file_by_state_raw_last_row.index.values)
csv_case_states_new = csv_case_states[csv_case_states['date'] >= startDate]

for i in range((int)((endDate - startDate) / np.timedelta64(1, "D")) + 1):
    date = startDate + np.timedelta64(i, "D")
    newCases = [csv_case_states_new[(csv_case_states_new['date'] == date) & (csv_case_states_new['state'] == state)]['cases_new'].values[0] for state in inputStateLabel]
    cases = np.add(newCases, lastCases)
    cumulNewCases = [cumulAndNewCases2Cell(cases[i], newCases[i]) for i in range(len(outputStateColumn))]
    file_by_state_raw_append.loc[i] = dict(zip(outputStateColumn, cumulNewCases))
    file_by_state_raw_append.loc[i, 'Date'] = date.strftime("%d/%m/%Y")
    file_by_state_raw_append.loc[i, 'Source'] = file_case_states
    lastCases = cases
file_by_state_raw_append.to_csv(rawByStateCSVFilename, mode = "a", header = False, index = False)
createByStateData(rawCSVFilename = rawByStateCSVFilename)

print("Finished!")


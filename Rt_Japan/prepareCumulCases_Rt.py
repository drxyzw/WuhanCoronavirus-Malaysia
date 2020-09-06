import pandas as pd
import numpy as np
from datetime import timedelta, datetime

url = 'data/JapaneseDataCOVID19 (200510).csv'
columns = ['onset', 'exp_type']
df = pd.read_csv(url, parse_dates=['onset'], usecols=columns)[columns]
df.dropna(subset=["onset"], inplace=True)

startDate = df['onset'].min()
endDate = df['onset'].max()
n = (endDate - startDate).days
dates = []
total = []
domestic = []
imported = []
for date in (startDate + timedelta(i) for i in range(n)):
    total_i = df[df['onset']==date].count()['onset']
    imported_i = df[(df['onset']==date) & (df['exp_type']=="imported")].count()['onset']
    domestic_i = df[(df['onset']==date) & (df['exp_type']=="domestic")].count()['onset']
    try:
        assert total_i == imported_i + domestic_i
    except AssertionError:
        print("total == imported + domestic does not hold")
    dates.append(date)
    total.append(total_i)
    imported.append(imported_i)
    domestic.append(domestic_i)

dfSave = pd.DataFrame(index = dates, data = {'onset': total, 'domestic': domestic, 'imported': imported})
dfSave.to_csv("data/JapanDailyOnset.csv", date_format='%Y-%m-%d')

# save onset data
state = len(dates) * ["Japan"]
dfIndex = pd.DataFrame({'state': state, 'date': dates})
dfForRt_dom = pd.DataFrame(index = pd.MultiIndex.from_frame(dfIndex), data = {'cases': domestic})
dfForRt_dom.to_csv("data/onsetForRt_dom.csv", date_format='%Y-%m-%d')
dfForRt = pd.DataFrame(index = pd.MultiIndex.from_frame(dfIndex), data = {'cases': total})
dfForRt.to_csv("data/onsetForRt.csv", date_format='%Y-%m-%d')

# save confirmed-only (no onset date) data
columns = ['confirmed', 'onset', 'exp_type']
dfRaw = pd.read_csv(url, parse_dates=['onset', 'confirmed'], usecols=columns)[columns]
dfRaw.dropna(subset=["confirmed"], inplace=True)
df = dfRaw[np.isnat(dfRaw['onset'])]
df = df[['confirmed', 'exp_type']]
dates = []
total = []
domestic = []
imported = []
for date in (startDate + timedelta(i) for i in range(n)):
    total_i = df[df['confirmed']==date].count()['confirmed']
    imported_i = df[(df['confirmed']==date) & (df['exp_type']=="imported")].count()['confirmed']
    domestic_i = df[(df['confirmed']==date) & (df['exp_type']=="domestic")].count()['confirmed']
    try:
        assert total_i == imported_i + domestic_i
    except AssertionError:
        print("total == imported + domestic does not hold")
    dates.append(date)
    total.append(total_i)
    imported.append(imported_i)
    domestic.append(domestic_i)
# save confirmed only data
state = len(dates) * ["Japan"]
dfIndex = pd.DataFrame({'state': state, 'date': dates})
dfForRt_dom = pd.DataFrame(index = pd.MultiIndex.from_frame(dfIndex), data = {'cases': domestic})
dfForRt_dom.to_csv("data/confirmedOnlyForRt_dom.csv", date_format='%Y-%m-%d')
dfForRt = pd.DataFrame(index = pd.MultiIndex.from_frame(dfIndex), data = {'cases': total})
dfForRt.to_csv("data/confirmedOnlyForRt.csv", date_format='%Y-%m-%d')

print("Done");

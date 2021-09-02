# Based on R sctipt: https://gist.github.com/wnarifin/a608e60b6d35fdb369ee8133b30d36ab

from datetime import datetime
from datetime import timedelta
import requests
import urllib.request
from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv
import re
from dataCreateByState import createByStateData
from dataCreateByState import cell2TwoNumbersByBracket
from dataCreateTotal import createTotalData

stateNames = [
    ["SABAH", "SB"],
    ["SELANGOR", "SE"],
    [["W.P. KUALA LUMPUR", "WP KUALA LUMPUR"], "KL"],
    ["KEDAH", "KD"],
    ["NEGERI SEMBILAN", "NS"],
    ["PULAU PINANG", "PG"],
    ["SARAWAK", "SR"],
    [["W.P. LABUAN", "WP LABUAN"], "LB"],
    ["JOHOR", "JH"],
    ["PERAK", "PK"],
    ["PAHANG", "PH"],
    ["MELAKA", "ML"],
    ["TERENGGANU", "TR"],
    ["KELANTAN", "KE"],
    [["W.P. PUTRAJAYA", "WP PUTRAJAYA"], "PT"],
    ["PERLIS", "PR"]
    ]

statesDestinationOrdered = [
        "JH",
        "KD",
        "KE",
        "ML",
        "NS",
        "PH",
        "PG",
        "PK",
        "PR",
        "SB",
        "SR",
        "SE",
        "TR",
        "KL",
        "PT",
        "LB"
    ]

months_malay = ["januari", "februari", "mac", "april", "mei", "jun", "julai",    "ogos", "september", "oktober", "november", "disember"]

def getHealthDirectorBlogHtml(thisdate, isToday):
    y = thisdate.strftime("%Y")
    m = thisdate.strftime("%m")
    m_int = thisdate.month
    m_malay = months_malay[m_int-1]
    d = thisdate.strftime("%d")
    d_int = thisdate.day
    thisdate_minus1 = thisdate + timedelta(days = -1)
    y_m1 = thisdate_minus1.strftime("%Y")
    m_m1 = thisdate_minus1.strftime("%m")
    m_int_m1 = thisdate_minus1.month
    m_malay_m1 = months_malay[m_int_m1-1]
    d_m1 = thisdate_minus1.strftime("%d")
    d_int_m1 = thisdate_minus1.day
    
    url = "https://kpkesihatan.com/" + y + "/" + m + "/" + d + "/kenyataan-akhbar-kpk-" + str(d_int) + "-" + m_malay + "-" + y  + "-situasi-semasa-jangkitan-penyakit-coronavirus-2019-covid-19-di-malaysia/"
    suburl = "https://kpkesihatan.com/" + y + "/" + m + "/" + d + "/kenyataan-akhbar-" + str(d_int) + "-" + m_malay + "-" + y + "-2020-situasi-semasa-jangkitan-penyakitcoronavirus-2019-covid-19-di-malaysia/"
    url2 = "https://kpkesihatan.com/" + y + "/" + m + "/" + d + "/kenyataan-akhbar-kpk-" + str(d_int_m1) + "-" + m_malay + "-" + y_m1  + "-situasi-semasa-jangkitan-penyakitcoronavirus-2019-covid-19-di-malaysia-2/"
    url_ybmk = "https://kpkesihatan.com/" + y + "/" + m + "/" + d + "/kenyataan-akhbar-ybmk-" + str(d_int) + "-" + m_malay + "-" + y  + "-situasi-semasa-jangkitan-penyakit-coronavirus-2019-covid-19-di-malaysia/"
    # check if url exists
    request_res = requests.head(url)
    if(request_res.status_code != 200):
        request_res = requests.head (suburl)
        if(request_res.status_code != 200):
            request_res = requests.head (url2)
            if(request_res.status_code != 200):
                request_res = requests.head (url_ybmk)
                if(request_res.status_code != 200):
                    if(isToday):
                        return None, None
                    else:
                        raise Exception("none of urls exists  or is valid: " + url + ", " + suburl + "," + url2 + "," + url_ybmk)
                else:
                    url = url_ybmk
            else:
                url = url2
        else:
            url = suburl
    html = urlopen(url)
    return html, url

today = datetime.today()
# load nationwide data
rawTotalCSVFilename = './data/totalRaw.csv'
headers = ["Date", "Total cases", "New cases", "Total deaths", "Total recovered"]
with open(rawTotalCSVFilename, 'r+', newline='', encoding='utf-8') as csvfile: # read and append, and file cursor at the beginning
    lines = list(csv.reader(csvfile))
    lastDateStr = lines[-1][0]
    lastDate = datetime.strptime(lastDateStr, "%Y-%m-%d")
    startdate = lastDate + timedelta(days = 1)
    enddate = today
    l = (enddate - startdate).days + 1
    outputRows = []
    outputHeader = lines[0]
    for i in range(l):
        outputRow = [""] * len(outputHeader)
        thisdate = startdate + timedelta(days = i)
        html, url = getHealthDirectorBlogHtml(thisdate, i == l - 1)
        if i == l - 1 and html == None:
            break
        soup = BeautifulSoup(html,  'html.parser')
        lists = soup.find_all('li')
        cases = re.findall(r'\d*\.?\d+',[s for s in lists if "Kes baharu" in s.text][0].text.replace(',',''))
        total_cases = cases[1]
        new_cases = cases[0]
        total_deaths = re.findall(r'\d*\.?\d+',[s for s in lists if "Kes kematian" in s.text][0].text.replace(',',''))[1]
        total_rec = re.findall(r'\d*\.?\d+',[s for s in lists if "Kes sembuh" in s.text or "Kes   sembuh" in s.text][0].text.replace(',',''))[1]
        outputRow[0] = str(thisdate.year) + "-" + str(thisdate.month).zfill(2) + "-" + str(thisdate.day).zfill(2)
        outputRow[1] = str(total_cases)
        outputRow[2] = str(new_cases)
        outputRow[3] = str(total_deaths)
        outputRow[4] = str(total_rec)
        outputRows.append(outputRow)
    writer = csv.writer(csvfile)
    writer.writerows(outputRows)
reader = csv.reader(open(rawTotalCSVFilename, "r"), delimiter=",")
totalTable = list(reader)
createTotalData(totalTable)

# load by-state data
rawByStateCSVFilename = './data/byStatesRaw.csv'
with open(rawByStateCSVFilename, 'r+', newline='', encoding='utf-8') as csvfile: # read and append, and file cursor at the beginning
    lines = list(csv.reader(csvfile))
    lastDateStr = lines[-1][0]
    lastDate = datetime.strptime(lastDateStr, "%d/%m/%Y")

    startdate = lastDate + timedelta(days = 1)
    enddate = today
    l = (enddate - startdate).days + 1
    outputRows = []
    outputHeader = ["Date"] +   statesDestinationOrdered + ["Source"]
    
    for i in range(l):
        thisdate = startdate + timedelta(days = i)
        html, url = getHealthDirectorBlogHtml(thisdate, i == l - 1)
        if i == l - 1 and html == None:
            break
        soup = BeautifulSoup(html,  'html.parser')
        tables = soup.findAll("figure", {"class": "wp-block-table"})
        stateNames_full, stateNames_ab = zip(*stateNames)
    
        for table in tables:
            tableTag = table.find('table')
            if tableTag != None:
                tableBody = tableTag.tbody if (tableTag.find('tbody') != None) else tableTag

                # loading by-state table
                headers = tableBody.findAll('td')
                # up to 10 Aug 2021
                if headers[0].text == 'NEGERI' and headers[1].text.startswith("BILANGAN KES BAHARU") and headers[2].text == "BILANGAN KES KUMULATIF":
                    outputRow = []
                    y = thisdate.strftime("%Y")
                    m_int = thisdate.month
                    d_int = thisdate.day
                    dateStr = str(d_int) + "/" + str(m_int) + "/" + y
                    outputRow.append(dateStr)
                    for state in statesDestinationOrdered:
                        i = stateNames_ab.index (state)
                        stateName_full = stateNames_full[i]
                        for row in tableBody.findAll("tr"):
                            cells = row.findAll ("td")
                            cellText = str.strip(cells[0].text)
                            cellText = ' '.join(cellText.split())
                            if (type(stateName_full) is str and cellText == stateName_full) or (type(stateName_full) is list and any(x == cellText for x in stateName_full)):
                                newCases, dummy = cell2TwoNumbersByBracket(str.strip(cells  [1].text)) # new case might be a form of "[total new cases] ([import     cases]"
                                newCases = str(newCases)
                                cumulCases = str.strip(cells[2].text)

                                # when blog post number is incorrect
                                if d_int == 7 and m_int == 1 and y == "2021" and state == "KL":
                                    cumulCases = "15,258"

                                combinedCases = (cumulCases + "(" + newCases + ")") if newCases != "0" else cumulCases
                                outputRow.append(combinedCases)
                    outputRow.append(url) # Source column
                    outputRows.append(outputRow)

                # 11 Aug 2021 to 21 Aug 2021
                if headers[0].text == 'NEGERI' and headers[3].text.startswith("BILANGAN KES BAHARU") and headers[4].text == "BILANGAN KES KUMULATIF":
                    outputRow = []
                    y = thisdate.strftime("%Y")
                    m_int = thisdate.month
                    d_int = thisdate.day
                    dateStr = str(d_int) + "/" + str(m_int) + "/" + y
                    outputRow.append(dateStr)
                    for state in statesDestinationOrdered:
                        i = stateNames_ab.index (state)
                        stateName_full = stateNames_full[i]
                        for row in tableBody.findAll("tr"):
                            cells = row.findAll ("td")
                            cellText = str.strip(cells[0].text)
                            cellText = ' '.join(cellText.split())
                            if (type(stateName_full) is str and cellText == stateName_full) or (type(stateName_full) is list and any(x == cellText for x in stateName_full)):
                                newCases, dummy = cell2TwoNumbersByBracket(str.strip(cells[3].text)) # new case might be a form of "[total new cases] ([import cases]"
                                newCases = str(newCases)
                                cumulCases = str.strip(cells[4].text)

                                # when blog post number is incorrect
                                #if d_int == 7 and m_int == 1 and y == "2021" and state == "KL":
                                #    cumulCases = "15,258"

                                combinedCases = (cumulCases + "(" + newCases + ")") if newCases != "0" else cumulCases
                                outputRow.append(combinedCases)
                    outputRow.append(url) # Source column
                    outputRows.append(outputRow)
    
    writer = csv.writer(csvfile)
    writer.writerows(outputRows)

createByStateData(rawCSVFilename = rawByStateCSVFilename)
print("Finished")

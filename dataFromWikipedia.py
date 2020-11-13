import requests
import csv
import urllib.request
from bs4 import BeautifulSoup
from urllib.request import urlopen
from dataCreateByState import createByStateData

def getWebTableByCaption2CSV(url, captionWanted, filename):
    html = urlopen(url)

    soup = BeautifulSoup(html, 'html.parser')
    for caption in soup.find_all("caption"):
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

url = "https://en.wikipedia.org/wiki/Timeline_of_the_COVID-19_pandemic_in_Malaysia#Statistics"
#url = "https://en.wikipedia.org/wiki/2020_coronavirus_pandemic_in_Malaysia#Statistics"
# find a table with "Distribution of cumulative confirmed cases in various administrative regions of Malaysia"
captionWanted = "Distribution of cumulative confirmed cases in various administrative regions of Malaysia"
rawCSVFilename = './data/byStatesRaw.csv'
getWebTableByCaption2CSV(url, captionWanted, rawCSVFilename)

createByStateData(rawCSVFilename = rawCSVFilename)

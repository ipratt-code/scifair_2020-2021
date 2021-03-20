# this is a short and simple script
# for taking the total deaths and putting them in a more human readable format
# make sure you are running this script inside the output-data directory
# or else it won't work

import pandas as pd

countries = ["South Korea", "United States"]
diseases = ["1918h1n1", "coronavirus", "measles", "smallpox"]

files = [cnt + "_" + dis + ".csv" for cnt in countries for dis in diseases]
filesToCountries = dict()
for cnt in countries:
    for dis in diseases:
        filesToCountries[cnt + "_" + dis + ".csv"] = cnt
# print(files)
skor_deaths = []
usa_deaths = []
for datafile in files:
    df = pd.read_csv(datafile)
    if filesToCountries[datafile] == "South Korea":
        skor_deaths.append(df.iloc[-1, -1])
    elif filesToCountries[datafile] == "United States":
        usa_deaths.append(df.iloc[-1, -1])
    # print(df.iloc[-1, -1])
print(skor_deaths)
print(usa_deaths)
# this is a short and simple script
# for taking the total deaths and putting them in a more human readable format

import pandas as pd

countries = ["South Korea", "United States"]
diseases = ["1918h1n1", "coronavirus", "measles", "smallpox"]

files = [cnt + "_" + dis + ".csv" for cnt in countries for dis in diseases]
filesToCountries = dict()
for cnt in countries:
    for dis in diseases:
        filesToCountries[cnt + "_" + dis + ".csv"] = cnt

total_deaths = []
for datafile in files:
    df = pd.read_csv("output-data/" + datafile)
    total_deaths.append(list(df.iloc[-1, :])[25::25])

out_dict = {}
for i in range(len(total_deaths)):
    name = files[i][:-4]
    deaths = total_deaths[i]
    out_dict[name] = deaths

# print(out_dict)
out_df = pd.DataFrame.from_dict(out_dict)

out_df.index = [
    25,
    50,
    75,
    100,
    125,
    150,
    175,
    200,
    225,
    250,
    275,
    300,
    325,
    350,
    375,
    400,
    425,
    450,
    475,
    500,
]

print(out_df)

out_df.to_csv("output-data/total-deaths-time-slices-all-simulations.csv")
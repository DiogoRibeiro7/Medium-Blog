import os
import pandas as pd
import re
from difflib import SequenceMatcher
import seaborn
import sqlite3 as sqlite

# ------------------------------------------------------------------------------------------ #
# Food inspection data in Las Vegas
# ------------------------------------------------------------------------------------------ #


filepath = “/FILE_PATH/“

FoodInsp = pd.read_csv(os.path.join(filepath, 'Restaurant_Inspections_LV.csv'), header = 0)
FoodInsp1 = FoodInsp.loc[FoodInsp['City']=='Las Vegas'] # consider obs only in Las Vegas

# min(FoodInsp['Inspection Date'])
# max(FoodInsp['Inspection Date'])

FoodInsp2 = FoodInsp1.groupby('Permit Number').last()  # keep the latest inspection result; Permit Number was sorted by the date already.
FoodInsp2 = FoodInsp2.reset_index()

# FoodInsp2.columns
# FoodInsp2.shape # 15157


# ------------------------------------------------------------------------------------------ #
# Yelp data
# ------------------------------------------------------------------------------------------ #
# ref : http://stackoverflow.com/questions/30088006/cant-figure-out-how-to-fix-the-error-in-the-following-code
# read the entire file into a python array
with open(os.path.join(filepath, 'yelp_dataset_challenge_academic_dataset/yelp_academic_dataset_business.json'), 'rb') as f:
    data = f.readlines()

# remove the trailing "\n" from each line
data = map(lambda x: x.rstrip(), data)
data_json_str = "[" + ','.join(data) + "]"
data_df = pd.read_json(data_json_str)

Yelp = data_df

# FoodInsp.shape  # (113416, 23)
# Yelp.shape  # (85901, 15)

Yelp['full_address'] = Yelp['full_address'].apply(lambda x: x.replace('\n', ' '))
Yelp1 = Yelp.loc[(Yelp['state']=='NV') & (Yelp['city']=='Las Vegas')] # 19326
Yelp1 = Yelp1[['full_address', 'latitude', 'longitude', 'name','review_count', 'stars','categories']]
Yelp1.shape  # 19326

Zip_code_Yelp = pd.DataFrame(Yelp1['full_address'].str.extract('.+\s(\d{5}|\d{5}\-\d{4})', expand=True))
Zip_code_Yelp.columns = ['zip']

if 'zip' in Yelp1.columns: del Yelp1['zip']
Yelp1 = Yelp1.join(Zip_code_Yelp)

Rest = Yelp1.apply(lambda row: True if ('Restaurants' in row['categories']) or ('Bars' in row['categories']) or ('Food' in row['categories']) else False, axis=1)
Yelp4 = Yelp1.loc[Rest]  # only include restaurant, bar and food shops
# Yelp4.shape # 6881


# ------------------------------------------------------------------------------------------ #
# Build up pair of indices (Yelp_index, Food_Insp_index) in order to use join using relational database tables
#
# Based on Yelp data, for each observation's name,
# constrain the food inspection data which have the same zip code and
# compare the similarity of the shop name using SequenceMatcher function in 'matchingStore' (User defined function; UDF)
# select the index(indices) if the similarity >= 90
#
# No matched cases are dropped.
#
# Make pairs of indices (Yelp_index, Food_Insp_index) that are possible to be 1-to-many relationship.
# Using relational database, create the third table consisting of these pairs of indices.
# Use left join to combine two dataset based on primary key (index).
# After joining, validate the data by comparing the addresses that were from both dataset.
# If the starting digits (street number) are matched, then the obs was kept. Otherwise, dropped.
# ------------------------------------------------------------------------------------------ #

# Function which returns a ratio how two strings are similar
# from module <difflib>
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

# in order to use similar function, make full_address (Yelp) into short address (drop city, state, zipcode, USA)
shortaddr = pd.DataFrame(Yelp4['full_address'].apply(lambda x : re.findall(r'(.+)\sLas\sVegas\,\sNV\s([\d]{5}).*', x)))
shortaddr.rename(columns={'full_address': 'shortaddr'}, inplace=True)
if 'shortaddr' in Yelp4.columns: del Yelp4['shortaddr']
Yelp4 = Yelp4.join(shortaddr)
Yelp4['ShortAddr'] = Yelp4['shortaddr'].apply(lambda x: x[0][0] if len(x) > 0 else None)
Yelp4['ShortZip'] = Yelp4['shortaddr'].apply(lambda x: x[0][1] if len(x) > 0 else None)
del Yelp4['shortaddr']

# Yelp4[Yelp4['zip']!=Yelp4['ShortZip']].shape  # 42 -> very short address and no zip in it -> ignore
Yelp4 = Yelp4[Yelp4['zip']==Yelp4['ShortZip']]  # (6839, 10)

# To show the completion status of generating the pair of indices, make a separate column consisting of integer sequence.
status_num = pd.DataFrame(range(1, Yelp4.shape[0]+1), columns = ['status_bar_int'], index=Yelp4.index).astype(int)
if 'status_bar_int' in Yelp4.columns: del Yelp4['status_bar_int']
Yelp4 = Yelp4.join(status_num)

Yelp5 = Yelp4
# Yelp5.shape  # (6839, 11)
Yelp5_row_num = Yelp5.shape[0]

# extract first 5 digis from zip code in Food Inspection dataset
FoodInsp2['zip'] = FoodInsp2['Zip'].astype(str).apply(lambda x : x[0:5])




# within the same zip code find the matched restaurant name
def matchingStore(refname, zipcode, status_int):
    # constrain in the same zip code area to speed up for matching
    df = FoodInsp2.loc[FoodInsp2['zip']==zipcode]

    # compare the similarity between store name from Yelp and from Food inspection data
    testratio = df['Restaurant Name'].apply(lambda x: True if similar(refname.lower(), x.lower()) >= 0.9 else False)

    # print the completion status
    print ("{:10.2f}".format(float(status_int*100)/Yelp5_row_num) + '% is completed!')

    return df[testratio].index.tolist()



# takes about 20 mins
testMatching = pd.DataFrame(Yelp5.apply(lambda row: matchingStore(row['name'], row['ShortZip'], row['status_bar_int']), axis=1), columns=['matchedIdx'])

if 'matchedIdx' in Yelp5.columns: del Yelp5['matchedIdx']
Yelp5 = pd.DataFrame(Yelp5.join(testMatching))

Yelp5['len'] =Yelp5['matchedIdx'].apply(lambda x : len(x))
# Yelp5.shape # (6839, 13)

Yelp6 = Yelp5[Yelp5['len']!=0]  # filter out no matched case
# Yelp6.shape  # (1570, 13)
# Yelp6.columns




# genrate a list of tuples. Tuples are the pair of Yelo_index and Food_Insp_index
def matchingIdx(row):
    idxList = row['matchedIdx']
    idxYelp = row.name
    idxtup = []

    for idx in idxList:
        idxtup.append([idxYelp, idx])

    return idxtup

Idx_Pair_ListOfList = Yelp6.apply(matchingIdx, axis = 1)
Idx_Pair = [tuple(item) for pair in Idx_Pair_ListOfList for item in pair]  # [ ..., (Yelp_idx, Food_Insp_idx),...]



# create index column separately for each dataset
Yelp6['Idx_Yelp'] = Yelp6.index
FoodInsp2['Idx_FoodInsp'] = FoodInsp2.index



# convert Yelp & Food_Insp df into DB table and using primary key, do left join
# ref : http://stackoverflow.com/questions/36593552/saving-a-pandas-dataframe-into-sqlite-with-different-column-names

with sqlite.connect(os.path.join(filepath,'Yelp_Food_idx_pair.db')) as con:
    cur = con.cursor()

    # Yelp data
    cur.execute("DROP TABLE IF EXISTS Yelp")
    cur.execute("CREATE TABLE Yelp(Idx INT, Addr TEXT, ShortAddr TEXT, lat REAL, long REAL, name TEXT, stars REAL)")
    cur.executemany("INSERT INTO Yelp (Idx, Addr, ShortAddr, lat, long, name, stars) VALUES(?,?,?,?,?,?,?)",list(Yelp6[['Idx_Yelp', 'full_address', 'ShortAddr', 'latitude', 'longitude', 'name', 'stars']].to_records(index=False)))

    # Food Inspection data
    cur.execute("DROP TABLE IF EXISTS FoodInsp")
    cur.execute("CREATE TABLE FoodInsp(Idx INT, Rest_name TEXT, Addr TEXT, Insp_Date TEXT, Insp_Grade TEXT)")
    cur.executemany("INSERT INTO FoodInsp(Idx, Rest_name, Addr, Insp_Date, Insp_Grade) values(?,?,?,?,?)", list(FoodInsp2[['Idx_FoodInsp', 'Restaurant Name', 'Address', 'Inspection Date', 'Inspection Grade']].to_records(index=False)))

    # primary key table : pair of indices
    cur.execute("DROP TABLE IF EXISTS IdxPair")
    cur.execute("CREATE TABLE IdxPair(Yelp_Idx INT, FoodInsp_Idx INT)")
    cur.executemany("INSERT INTO IdxPair(Yelp_Idx, FoodInsp_Idx) values(?,?)", Idx_Pair)

    # Left join
    cur.execute("DROP TABLE IF EXISTS Yelp_FoodInsp")
    cur.execute("CREATE TABLE Yelp_FoodInsp AS SELECT y.Idx AS Yelp_idx, f.Idx AS FoodInsp_idx, y.name, f.Rest_name, y.ShortAddr, y.Addr, f.Addr AS Addr_FoodInsp, y.lat, y.long, f.Insp_Date, f.Insp_Grade, y.stars FROM Yelp y LEFT JOIN IdxPair ip ON y.Idx = ip.Yelp_Idx LEFT JOIN FoodInsp f ON ip.FoodInsp_Idx = f.Idx")

    con.commit()

    # the combined dataset is read into dataframe
    # ref : http://stackoverflow.com/questions/10065051/python-pandas-and-databases-like-mysql
    sql = "SELECT * FROM Yelp_FoodInsp"
    Yelp_Food3 = pd.read_sql_query(sql, con)


# combined dataset
# Yelp_Food3.shape # (1706, 12)


# validate the matching by comparing their similarity or the first digits of stree address
def Ind(row):
    if similar(row['ShortAddr'], row['Addr_FoodInsp']) >= 0.9: # if the addresses are similar already, keep the row
        return 1

    else: # if the address are not similar, take the first digits of address and compare it
        if re.findall(r'^([\d]{1,})\s.+', row['ShortAddr']) == re.findall(r'^([\d]{1,})\s.+', row['Addr_FoodInsp']):
            return 1
    return 0

Yelp_Food = Yelp_Food3
Indicator = pd.DataFrame(Yelp_Food.apply(Ind, axis=1), columns=['Ind'])

if 'Ind' in Yelp_Food.columns: del Yelp_Food['Ind']
Yelp_Food1 = Yelp_Food.join(Indicator)

# keep the observations if they are valid (same name/address/zip)
Yelp_Food2 = Yelp_Food1.loc[Yelp_Food1['Ind']==1]
# Yelp_Food2.shape  # (1197, 40) -> (1328, 13)


# give the numeric value for Inspection grade in order to be used for different coloring
g = Yelp_Food2.groupby(['Insp_Grade'])
# g.size()
# grade N(1), O (1), X (8) -> drop

def Grade_into_num(grade):
    if grade == 'A': return 5
    if grade == 'B': return 4
    if grade == 'C': return 3
    return 0

# meaning of grade (ordinal variable)
# ref : http://www.southernnevadahealthdistrict.org/ferl/grade-card.php
# 'A' indicates that a restaurant received 0-10 demerits on their last inspection.
# 'B' indicates between 11 and 20 demerits or identical consecutive critical or major violations.
# 'C' indicatesbetween 21 and 40 demerits, has identical consecutive critical or major violations or more than 10 demerits on a ‘B’ grade re-inspection.

GradeNum = pd.DataFrame(Yelp_Food2['Insp_Grade'].apply(lambda x: Grade_into_num(x)))
GradeNum.rename(columns={'Insp_Grade': 'Grade stars'}, inplace=True)

if 'Grade stars' in Yelp_Food2.columns: del Yelp_Food2['Grade stars']
Yelp_Food2 = Yelp_Food2.join(GradeNum)

# Yelp_Food2.shape  # 1328
Yelp_Food2['visual_ind'] = Yelp_Food2['Insp_Grade'].apply(lambda x : True if ((x=='A') or (x=='B') or x=='C') else False)
# Yelp_Food2.loc[Yelp_Food2['visual_ind']==True].shape  # drop N, O and X -> 1317

Yelp_Food2 = Yelp_Food2.loc[Yelp_Food2['visual_ind']==True]  # keep the record only if it has a food inspection grade

# ------------------------------------------------------------------------------------------ #
# Visualization
# ------------------------------------------------------------------------------------------ #

# 1. joint plot
# ref : https://www.dataquest.io/blog/python-data-visualization-libraries/

data = pd.DataFrame({"grade": Yelp_Food2["Grade stars"], "stars": Yelp_Food2["stars"]})
seaborn.jointplot(x="stars", y="grade", data=data)
# hard to decide whether there is a linear relationship b/w two variables since R^2 = 0.61


# 2. In order to do Fisher exact test for investigating their relationship and another visualization (interactive map),
# The combined dataset was write out as csv file
# that consists of
#   - coordinates of lat & long
#   - Restaurant name
#   - Yelp rating stars
#   - Food Inspection Result (Grade) and its date

# take only date part in datetime format variable
Date = pd.DataFrame(Yelp_Food2['Insp_Date'].apply(lambda x : re.findall(r'^([\d]{2}\/[\d]{2}\/[\d]{4})\s.+',x)[0]))
Date.rename(columns={'Insp_Date': 'Date'}, inplace=True)
if 'Date' in Yelp_Food2.columns: del Yelp_Food2['Date']
Yelp_Food2 = Yelp_Food2.join(Date)

# min(Yelp_Food2['Date']) # 01/03/2014
# max(Yelp_Food2['Date']) # 12/31/2013

# make a text which will be displayed on the map in R
if 'text' in Yelp_Food2.columns: del Yelp_Food2['text']
Yelp_Food2['text'] = Yelp_Food2['name'] + '<br> stars: ' + Yelp_Food2['stars'].astype(str) + \
                     '<br>Food Inspection Result : ' + Yelp_Food2['Insp_Grade'] + \
                    '<br>Inspection Date : ' + Yelp_Food2['Date']


# Yelp_Food2['text'] = Yelp_Food2['name']
mapDF = Yelp_Food2[['lat', 'long', 'text', 'stars', 'Grade stars']]
# mapDF.columns
# mapDF.shape # (1188, 5) -> (1317, 5)
mapDF.rename(columns={'lat': 'Latitude', 'long': 'Longitude', 'text': 'Text', 'stars': 'Yelp Stars', 'Grade stars': 'Inspection Grade'}, inplace=True)
# mapDF.groupby(['Inspection Grade']).size()
# # Inspection Grade
# # 3      22
# # 4      62
# # 5    1233


filepath_out = “OUTPUT_PATH”
mapDF.to_csv(os.path.join(filepath_out, 'Yelp_Inspection_comparison_output.csv'), encoding='utf-8', index=False)

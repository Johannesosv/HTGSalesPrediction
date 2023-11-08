# %%
#create XG Boost model with amount as target
import xgboost as xgb
import numpy as np
import pandas as pd
import datetime
import pyodbc
from sqlalchemy import event, create_engine, text
import urllib.parse
import pandas as pd
import sys

# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    config_path = "C:\\Users\\Jojje\\Desktop\\Project_Desktop\\sql config.txt"

# Access the argument
config_path = sys.argv[1]

# %%
#get swedish holidays
from workalendar.europe import Sweden
cal = Sweden()
#get min and max years from current year
min_year = datetime.datetime.now().year
max_year = datetime.datetime.now().year


#get holidays for each year
holidays = []
for year in range(min_year,max_year+1):
    holidays.append(cal.holidays(year))


#make df of holidays
holidays_df = pd.concat([pd.DataFrame(holiday) for holiday in holidays], ignore_index=True)


#rename columns
holidays_df = holidays_df.rename(columns={0:'date',1:'holiday'})


#make date to pandas datetime
holidays_df['date'] = pd.to_datetime(holidays_df['date'], errors='coerce')

# %%
#get todays date and make prediction for the next 7 days
todays_date = datetime.datetime.now().date()

#make df of next 7 days
date_list = []
for i in range(1,8):
    date_list.append(todays_date + datetime.timedelta(days=i))

date_df = pd.DataFrame(date_list)
date_df = date_df.rename(columns={0:'date'})

#make date to pandas datetime
date_df['date'] = pd.to_datetime(date_df['date'], errors='coerce')

#add the date of the previous holiday and the next holiday to SQL_df
date_df['prev_holiday'] = date_df['date'].apply(lambda x: holidays_df[holidays_df['date']<x]['date'].max())
date_df['next_holiday'] = date_df['date'].apply(lambda x: holidays_df[holidays_df['date']>x]['date'].min())

#Calculate days to previous and next holiday, the result should be in integer
date_df['days_to_prev_holiday'] = -(date_df['date'] - date_df['prev_holiday']).dt.days
date_df['days_to_next_holiday'] = (date_df['next_holiday'] - date_df['date']).dt.days

#take the value from the column that is closest in absolute days. Do not use min value to compare because it will return the negative value, but insert negative value to the column that is negative
date_df['days_to_holiday'] = date_df.apply(lambda row: row['days_to_prev_holiday'] if abs(row['days_to_prev_holiday']) < abs(row['days_to_next_holiday']) else row['days_to_next_holiday'], axis=1)

#drop columns
date_df = date_df.drop(['prev_holiday','next_holiday','days_to_prev_holiday','days_to_next_holiday'], axis=1)

#add columns
date_df['TheWeek'] = date_df['date'].dt.isocalendar().week
date_df['TheYear'] = date_df['date'].dt.isocalendar().year
date_df['IsWeekend'] = date_df['date'].dt.dayofweek.apply(lambda x: 1 if x > 4 else 0)

#make date to pandas datetime
date_df['date'] = pd.to_datetime(date_df['date'], errors='coerce')

# %% [markdown]
# 
# pmean
# [0.0]
# kg/m2/h
# t
# [-3.5]
# Cel
# vis
# [3.5]
# km
# ws
# [1.4]
# m/s
# 

# %%
import requests

#around Älvsbyn
lon="21"
lat="66.6"

url_temp = "https://opendata-download-metfcst.smhi.se/api/category/pmp3g/version/2/geotype/point/lon/"+lon+"/lat/"+lat+"/data.json"
#/api/category/pmp3g/version/2/geotype/point/lon/{longitude}/lat/{latitude}/data.json


r=requests.get(url_temp)
data=r.json()

parameters = ["ObsDate","ws","pmean","t","vis"]

#make empty df with the columns and a datetime column
df = pd.DataFrame(columns=parameters)


for j in range(len(data["timeSeries"])):
    #for the time series get all values for the parameters in the list
    obs_date = data["timeSeries"][j]["validTime"]
    data_parameters = data["timeSeries"][j]["parameters"]
    for i in range(len(data_parameters)):
        if data_parameters[i]["name"] in parameters:
            parameter_name = data_parameters[i]["name"]
            parameter_value = data_parameters[i]["values"]
    
            #make a df with the values
            df_temp = pd.DataFrame(parameter_value, columns=[parameter_name])
            #add the datetime column
            df_temp["ObsDate"] = obs_date
            #add the df to the main df
            df = pd.concat([df, df_temp], ignore_index=True)

#make the datetime column to pandas datetime
df["ObsDate"] = pd.to_datetime(df["ObsDate"], errors='coerce')

#merge all row with the same date to one row
df = df.groupby("ObsDate").mean().reset_index()



# %%
#only get time between 9 and 17
df = df[(df["ObsDate"].dt.hour >= 9) & (df["ObsDate"].dt.hour <= 17)]

#group the data by date and get the sum for pmean and the mean for the rest
df = df.groupby(df["ObsDate"].dt.date).agg({'ws': 'mean', 'pmean': 'sum', 't': 'mean', 'vis': 'mean'}).reset_index()

#take the vis times 1000
df["vis"] = df["vis"]*1000

#rename columns
df = df.rename(columns={"ObsDate":"date","ws":"vindhastighet_avg","pmean":"nederbörd","t":"temperatur_avg","vis":"sikt_avg"})

#make date to pandas datetime
df['date'] = pd.to_datetime(df['date'], errors='coerce')

#merge the df
date_df = date_df.merge(df, on="date", how="left")

# %%
#load model from Other/xg_reg.pickle.dat
import pickle
xg_reg = pickle.load(open("Other/xg_reg.pickle.dat", "rb"))

#make prediction
preds = xg_reg.predict(date_df[['TheWeek',"TheYear","IsWeekend","nederbörd","temperatur_avg","vindhastighet_avg","sikt_avg","days_to_holiday"]])

#make df of predictions
preds_df = pd.DataFrame(preds)
preds_df = preds_df.rename(columns={0:'salesamount'})
preds_df['date'] = date_df['date']
preds_df['date'] = pd.to_datetime(preds_df['date'], errors='coerce')
preds_df['date'] = preds_df['date'].dt.strftime('%Y-%m-%d')

#add the prediction to the date_df
date_df['salesamount'] = preds_df['salesamount']

#add current datetime to date_df as ReadInTime
date_df['ReadInTime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

# %%
#read in the sql config file and insert into a dict
f = open(config_path, "r")
sql_config = {}
for line in f:
    key, value = line.split(" = ")
    sql_config[key] = value.strip()

#Set sql configuration
server   = sql_config["sql_server"]
port     = sql_config["sql_port"]
database = "ML"
driver = "SQL+Server+Native+Client+11.0"


#Read in sql login from config file
sql_username = sql_config["sql_username"]
sql_password = sql_config["sql_password"]

# %%


#-------    Read from Pandas DF into SQL Table    -------#
#create engine
connection_string = f"mssql+pyodbc://{sql_username}:{sql_password}@{server},{port}/{database}?driver=SQL+Server+Native+Client+11.0"
engine = create_engine(connection_string)

@event.listens_for(engine, 'before_cursor_execute')
def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
    if executemany:
        cursor.fast_executemany = True
        cursor.commit()

date_df.to_sql("stg_ML_Results", engine, if_exists='append') #, if_exists='append', method='multi'

#Execute Stored Procedure
query = 'EXEC Load_ML_Results'
with engine.begin()as conn:
    result =  conn.execute(text(query))



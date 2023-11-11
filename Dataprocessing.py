# Dataprocessing.py
import datetime
import sys


import pandas as pd
import numpy as np

import pyodbc
from sqlalchemy import create_engine,event,text

#create XG Boost model with amount as target
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import seaborn as sns

from workalendar.europe import Sweden
import pickle
import requests



class DataProcessingAndModeling:
    def __init__(self, config_path):
        self.config_path = config_path
        self.data = None
        self.model = None
        self.model_parameters = {
            'learning_rate': 0.1,
            'max_depth': 4,
            'n_estimators': 100,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }

    def get_training_data_from_database(self):
        #read in the sql config file and insert into a dict
        f = open(self.config_path, "r")
        sql_config = {}
        for line in f:
            key, value = line.split(" = ")
            sql_config[key] = value.strip()

        #Set sql configuration
        server   = sql_config["sql_server"]
        port     = sql_config["sql_port"]
        database = "DIM"
        driver = "SQL+Server+Native+Client+11.0"


        #Read in sql login from config file
        sql_username = sql_config["sql_username"]
        sql_password = sql_config["sql_password"]

        #create engine
        connection_string = f"mssql+pyodbc://{sql_username}:{sql_password}@{server},{port}/{database}?driver=SQL+Server+Native+Client+11.0"
        engine = create_engine(connection_string)

        @event.listens_for(engine, 'before_cursor_execute')
        def receive_before_cursor_execute(conn, cursor, statement, params, context, executemany):
            if executemany:
                cursor.fast_executemany = True
                cursor.commit()
                
        self.data = pd.read_sql_query ('''
                                    SELECT
                                    *
                                    FROM [dbo].[VW_FactSales_ML] a
                                        ''', engine) 

        self.data['date'] = pd.to_datetime(self.data['date'], errors='coerce')
        
        #make all column names lower case
        self.data.columns = map(str.lower, self.data.columns)

        #remove  from clolumn names
        self.data.columns = self.data.columns.str.replace('_avg', '')

        print("Data is loaded with shape:",self.data.shape, "and columns:", self.data.columns.tolist())

    def get_prediction_data(self,num_days,lon,lat):
        #get todays date and make prediction for the next 7 days
        todays_date = datetime.datetime.now().date()

        #make df of next 7 days
        date_list = []
        for i in range(1,num_days+1):
            date_list.append(todays_date + datetime.timedelta(days=i))

        date_df = pd.DataFrame(date_list)
        date_df = date_df.rename(columns={0:'date'})
        date_df['date'] = pd.to_datetime(date_df['date'], errors='coerce')

        ##################
        #Get weather data
        ##################
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

        #take sikt times 1000 to get meters
        df['vis'] = df['vis']*1000

        #make the datetime column to pandas datetime
        df["ObsDate"] = pd.to_datetime(df["ObsDate"], errors='coerce').dt.date

        #merge all row with the same date to one row
        df = df.groupby("ObsDate").mean().reset_index()

        df["ObsDate"]=pd.to_datetime(df["ObsDate"], errors='coerce')
    
        df_tmp1 = pd.merge(date_df, df, left_on=date_df['date'], right_on=df['ObsDate'], how='left')

        #rename columns
        df_tmp1 = df_tmp1.drop(columns=['key_0','ObsDate'])
        df_tmp1 = df_tmp1.rename(columns={'ws':'vindhastighet', 'pmean':'nederbörd', 't':'temperatur', 'vis':'sikt'})
        self.data = df_tmp1

    def get_date_columns(self):
        #get year
        self.data['year'] = self.data['date'].dt.year
        #get month
        self.data['month'] = self.data['date'].dt.month
        #get monthname
        self.data['month_name'] = self.data['date'].dt.month_name().apply(lambda x: x[:3]).astype('category').cat.as_ordered()
        #get week
        self.data['week'] = self.data['date'].dt.isocalendar().week.astype('int64')
        #get day of month
        self.data['day_of_month'] = self.data['date'].dt.day
        #get day of week
        self.data['weekday_name'] = self.data['date'].dt.day_name().apply(lambda x: x[:3]).astype('category').cat.as_ordered()
        #make weekday_name only three letters and a categorical variable

        #Make cool varables
        #make lön_cat as a categorical variable with day 26-31 and 1-10 as the same category and the rest as the same category
        self.data['lön_cat'] = self.data['day_of_month'].apply(lambda x: 'Nära Löning' if x<=10 or x>=26 else 'Långt Från Löning').astype('category').cat.as_ordered()

        print("Date columns are loaded. Columns added: [month, month_name, week, day_of_month, day_of_week, lön_cat]")

    def get_holiday_data(self):
        #get swedish holidays
        cal = Sweden()

        #get min and max years from data date field
        min_year = self.data['date'].min().year
        max_year = self.data['date'].max().year


        #get holidays for each year
        holidays = []
        for year in range(min_year,max_year+1):
            holidays.append(cal.holidays(year))


        #make df of holidays
        holidays_df = pd.DataFrame()
        for i in range(len(holidays)):
            holidays_df = pd.concat([holidays_df, pd.DataFrame(holidays[i])], ignore_index=True)

        #rename columns
        holidays_df = holidays_df.rename(columns={0:'date',1:'holiday'})


        #make date to pandas datetime
        holidays_df['date'] = pd.to_datetime(holidays_df['date'], errors='coerce')

        #add the date of the previous holiday and the next holiday to self.data
        self.data['prev_holiday'] = self.data['date'].apply(lambda x: holidays_df[holidays_df['date']<=x]['date'].max())
        self.data['next_holiday'] = self.data['date'].apply(lambda x: holidays_df[holidays_df['date']>x]['date'].min())

        #Calculate days to previous and next holiday, the result should be in integer
        self.data['days_to_prev_holiday'] = -(self.data['date'] - self.data['prev_holiday']).dt.days
        self.data['days_to_next_holiday'] = (self.data['next_holiday'] - self.data['date']).dt.days

        #take the value from the column that is closest in absolute days. Do not use min value to compare because it will return the negative value, but insert negative value to the column that is negative
        self.data['days_to_holiday'] = self.data.apply(lambda row: row['days_to_prev_holiday'] if abs(row['days_to_prev_holiday']) < abs(row['days_to_next_holiday']) else row['days_to_next_holiday'], axis=1)

        #drop columns
        self.data = self.data.drop(['prev_holiday','next_holiday','days_to_prev_holiday','days_to_next_holiday'], axis=1)

        #make days to holiday categorical variable
        self.data['högtid_cat'] = self.data['days_to_holiday'].apply(lambda x: 'Högtid' if x == 0 else 'Före Högtid' if x > 0 and x<4 else 'Efter högtid' if x < 0 and x>-4 else 'Normal Dag').astype('category').cat.as_ordered()

        print("Holiday data is loaded. Columns added: [days_to_holiday, högtid_cat]")
        
    def make_weather_features(self):
        #make sikt a categorical variable
        self.data['sikt_cat'] = self.data['sikt'].apply(lambda x: 'Bra Sikt' if x>=25000 else 'Dålig Sikt').astype('category').cat.as_ordered()

        #make wind speed a categorical variable
        self.data['vindhastighet_cat'] = self.data['vindhastighet'].apply(lambda x: 'Stark vind' if x>=4 else 'Svag vind').astype('category').cat.as_ordered()

        #make nederbörd a categorical variable
        self.data['nederbörd_cat'] = self.data['nederbörd'].apply(lambda x: 'Regn' if x>5 else 'Lite Regn').astype('category').cat.as_ordered()

        #make temp a categorical variable depending on difference to the average temperature
        #insert average temperature per month manually
        average_temp_df = pd.DataFrame({'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                'Avg Temperature': [-11, -8, -1, 5, 12, 18, 20, 18, 12, 3, -3, -8]})

        #insert average temperature per month into self.data

        self.data = pd.merge(self.data, average_temp_df, left_on=self.data['month_name'], right_on=average_temp_df['Month'], how='left')
        self.data = self.data.drop(columns=['key_0'])

        self.data['temperatur_cat'] = self.data.apply(lambda row: 'Som Vanligt' if row['Avg Temperature']-3< row['temperatur'] and row['temperatur'] < row['Avg Temperature']+3 else 'Kallare än vanligt' if row['temperatur']<row['Avg Temperature']  else 'Varmare än vanligt', axis=1).astype('category').cat.as_ordered()

        #drop columns
        self.data = self.data.drop(['Avg Temperature','Month'], axis=1)

        print("weather features are loaded. Columns added: [sikt_cat, vindhastighet_cat, nederbörd_cat, temperatur_cat]")

    def train_model(self ,variable_list):
        #split data into train and test
        X = self.data[variable_list]
        y = self.data['salesamount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        #create model using the parameters in self.model_parameters

        self.model = xgb.XGBRegressor(objective ='reg:squarederror', enable_categorical=True, tree_method='hist', **self.model_parameters)

        #fit model
        self.model.fit(X_train,y_train)

        #make predictions on both the train and the test set and insert into self.data. Mark if the observation was in train or test set
        self.data['train_test'] = 'train'
        self.data.loc[X_test.index,'train_test'] = 'test'
        self.data['predicted_salesamount'] = self.model.predict(X)

        print("Model is trained and predictions are made. Columns added: [train_test, predicted_salesamount]")

        #calculate rmse
        rmse = np.sqrt(mean_squared_error(y_test, self.data.loc[X_test.index,'predicted_salesamount']))
        print("RMSE: %f" % (rmse))

        #calculate r2
        r2 = r2_score(y_test, self.data.loc[X_test.index,'predicted_salesamount'])
        print("R2: %f" % (r2))

    def get_best_model_param(self ,variable_list):
        #split data into train and test
        X = self.data[variable_list]
        y = self.data['salesamount']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

        # Assuming X_train and y_train are your training data
        param_grid = {
            'learning_rate': [0.01, 0.1, 0.2],
            'max_depth': [3, 4, 5],
            'n_estimators': [100, 200, 300],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0]
        }


        xgb_model = xgb.XGBRegressor(objective ='reg:squarederror', enable_categorical=True, tree_method='hist')

        grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, scoring='neg_mean_squared_error', cv=5)
        grid_result = grid_search.fit(X_train, y_train)

        # Print the best parameters
        print("Best Parameters: ", grid_result.best_params_)
        self.model_parameters = grid_result.best_params_


    def save_model(self, save_path):
        #save model
        pickle.dump(self.model, open(save_path, "wb"))
        print("Model is saved to:", save_path)

    def load_model(self, load_path):
        #load model
        self.model = pickle.load(open(load_path, "rb"))
        print("Model is loaded from:", load_path)
        
    def plot_features(self,variable_list):
        #if salesamount exists in self.data set variable to salesamount, else set to predicted_salesamount
        if 'salesamount' in self.data.columns:
            target = 'salesamount'
        else:
            target = 'predicted_salesamount'

        #print feature importance 
        xgb.plot_importance(self.model)
        plt.rcParams['figure.figsize'] = [5, 5]
        plt.show()

        # Create individual strip plots for each feature against salesamount, do not show y axis, make each category a different color
        plt.figure(figsize=(20, 6))
        for feature in variable_list:
            plt.subplot(2, 5, variable_list.index(feature) + 1)  # Adjust the subplot layout as needed
            sns.stripplot(x=feature, y=target, data=self.data, jitter=True, alpha=.25, size=3, palette='bright')
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.xticks(fontsize=9)
        plt.tight_layout()
        plt.suptitle('Distribution', fontsize=16)
        plt.show()

        #plot all features against the average sales amount per group of each category
        plt.figure(figsize=(20, 6))
        for feature in variable_list:
            plt.subplot(2, 5, variable_list.index(feature) + 1)  # Adjust the subplot layout as needed
            sns.barplot(x=feature, y=target, data=self.data)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.xticks(fontsize=9)
        plt.tight_layout()
        plt.suptitle('Distribution', fontsize=16)
        plt.show()

        #plot number of values in each category, add title to the whole plot
        plt.figure(figsize=(20, 6))
        for feature in variable_list:
            plt.subplot(2, 5, variable_list.index(feature) + 1)  # Adjust the subplot layout as needed
            sns.countplot(x=feature, data=self.data)
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.xticks(fontsize=9)
        plt.tight_layout()
        plt.suptitle('Number of values in each category', fontsize=16)
        plt.show()

    def make_predictions(self,varable_list):
        self.data["predicted_salesamount"] = self.model.predict(self.data[varable_list])
        print("Predictions are made. Columns added: [predicted_salesamount]")

    def get_predictions_to_database(self):
        #add current datetime to date_df as ReadInTime
        self.data['ReadInTime'] = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # %%
        #read in the sql config file and insert into a dict
        f = open(self.config_path, "r")
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

        self.data.to_sql("stg_ML_Results_new", engine, if_exists='append') #, if_exists='append', method='multi'

        #Execute Stored Procedure
        query = 'EXEC Load_ML_Results'
        with engine.begin()as conn:
            result =  conn.execute(text(query))

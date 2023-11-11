# %%
import datetime
import sys
import os
import pandas as pd
import numpy as np
import pyodbc
from sqlalchemy import create_engine,event
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from Dataprocessing import DataProcessingAndModeling

# %%
# Check if the correct number of arguments is provided
if len(sys.argv) != 2:
    config_path = "C:\\Users\\Jojje\\Desktop\\Project_Desktop\\sql config.txt"
else:
    config_path = sys.argv[1]
# Access the argument

current_path = os.getcwd()

variable_list = ["week",'weekday_name',"lön_cat","högtid_cat","nederbörd_cat","temperatur_cat","vindhastighet_cat","sikt_cat"]
model_path = current_path + "\\Other\\xg_reg.pickle.dat" 

# %%
#predict model
dataprocess = DataProcessingAndModeling(config_path)

prediction_days = 7
lon="21"
lat="66.6"

dataprocess.get_prediction_data(prediction_days,lon,lat)
dataprocess.get_date_columns()
dataprocess.get_holiday_data()
dataprocess.make_weather_features()
dataprocess.load_model(model_path)
dataprocess.make_predictions(variable_list)
dataprocess.get_predictions_to_database()

# %% [markdown]
# BELOW IS ANALYSIS





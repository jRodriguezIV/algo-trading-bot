import matplotlib.pyplot as plt
from arch import arch_model
import pandas as pd
import numpy as np
import os

data_folder = '.'

daily_df = pd.read_csv(os.path.join(data_folder, 'simulated_daily_data.csv')).dropna(axis=1)

daily_df['Date'] = pd.to_datetime(daily_df['Date'])

daily_df = daily_df.set_index('Date')

daily_df['log_ret'] = np.log(daily_df['Adj Close']).diff()
# daily_df 

intraday_5min_df = pd.read_csv(os.path.join(data_folder, 'simulated_5min_data.csv'))
intraday_5min_df['datetime'] = pd.to_datetime(intraday_5min_df['datetime'])
intraday_5min_df = intraday_5min_df.set_index('datetime')
intraday_5min_df['date'] = intraday_5min_df.index.strftime('%Y-%m-%d')
# intraday_5min_df
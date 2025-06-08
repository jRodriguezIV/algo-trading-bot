import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import datetime as dt
import yfinance as yf
import os

plt.style.use('ggplot')
data_folder = '.'

sentiment_df = pd.read_csv(os.path.join(data_folder, 'sentiment_data.csv'))
sentiment_df['date'] = pd.to_datetime(sentiment_df['date'])
sentiment_df = sentiment_df.set_index(['date', 'symbol'])
sentiment_df['engagement_ratio'] = sentiment_df['twitterComments']/sentiment_df['twitterLikes']
sentiment_df = sentiment_df[(sentiment_df['twitterLikes']>20)&(sentiment_df['twitterComments']>10)]
# sentiment_df

aggregated_df = (sentiment_df.reset_index('symbol').groupby([pd.Grouper(freq='ME'), 'symbol'])
                    [['engagement_ratio']].mean())
aggregated_df['rank'] = (aggregated_df.groupby(level=0)['engagement_ratio']
                        .transform(lambda x: x.rank(ascending=False)))
# aggregated_df

filtered_df = aggregated_df[aggregated_df['rank']<6].copy()
filtered_df = filtered_df.reset_index(level=1)
filtered_df.index = filtered_df.index+pd.DateOffset(1)
filtered_df = filtered_df.reset_index().set_index(['date', 'symbol'])
filtered_df = filtered_df.drop(['MRO'], level='symbol')
# filtered_df

dates = filtered_df.index.get_level_values('date').unique().tolist()
fixed_dates = {}

for d in dates:
    fixed_dates[d.strftime('%Y-%m-%d')] = filtered_df.xs(d, level=0).index.tolist()
# fixed_dates

stocks_list = sentiment_df.index.get_level_values('symbol').unique().tolist()
prices_df = yf.download(tickers=stocks_list, start='2021-01-01', end='2023-03-01')
prices_df = prices_df.drop(columns=['ATVI', 'MRO'], level='Ticker')
# prices_df

returns_df = np.log(prices_df['Close']).diff().dropna()
portfolio_df = pd.DataFrame()
for start_date in fixed_dates.keys():
    end_date = (pd.to_datetime(start_date)+pd.offsets.MonthEnd()).strftime('%Y-%m-%d')
    cols = fixed_dates[start_date]
    temp_df = returns_df[start_date:end_date][cols].mean(axis=1).to_frame('portfolio_return')
    portfolio_df = pd.concat([portfolio_df, temp_df], axis=0)
# portfolio_df
qqq_df= yf.download(tickers='QQQ',
                   start='2021-01-01',
                   end='2023-03-01')

qqq_ret = np.log(qqq_df['Close']).diff()

portfolio_df = portfolio_df.merge(qqq_ret,
                                 left_index=True,
                                 right_index=True)
portfolio_df = portfolio_df.rename(columns={'QQQ':'nasdaq_return'})
# portfolio_df

portfolios_cumulative_return = np.exp(np.log1p(portfolio_df).cumsum()).sub(1)

portfolios_cumulative_return.plot(figsize=(16,6))

plt.title('Twitter Engagement Ratio Strategy Return Over Time')

plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter(1))

plt.ylabel('Return')

plt.show()

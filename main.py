# Importing required modules

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#plt.style.use('fivethirtyeight')
#plt.show()

# Reading the data
stock_data = pd.read_csv('datasets/stock_data.csv',
                         parse_dates=['Date'],
                         index_col='Date'
                         ).dropna()
benchmark_data = pd.read_csv('datasets/benchmark_data.csv',
                             parse_dates=['Date'],
                             index_col='Date'
                             ).dropna()

# Displaying summary for stock_data
print('Stocks\n')
stock_data.info()
print(stock_data.head())

# Displaying summary for benchmark_data
print('\nBenchmarks\n')
benchmark_data.info()
benchmark_data.head()

# visualizing the stock_data
stock_data.plot(title='Stock Data', subplots=True)
plt.show()

# summarizing the stock_data
print(stock_data.describe())

# plotting the benchmark_data
benchmark_data.plot()
plt.show()

# summarizing the benchmark_data
print(benchmark_data.describe())

# calculating daily stock_data returns
stock_returns = stock_data.pct_change()

# plotting the daily returns
stock_returns.plot()
plt.show()

# summarizing the daily returns
print(stock_returns.describe())

# calculating daily benchmark_data returns
sp_returns = benchmark_data['S&P 500'].pct_change()

# plotting the daily returns
sp_returns.plot()
plt.show()

# summarizing the daily returns
print(sp_returns.describe())

# calculating the difference in daily returns
excess_returns = stock_returns.sub(sp_returns, axis=0)

# plotting the excess_returns
excess_returns.plot()
plt.show()

# summarizing the excess_returns
print(excess_returns.describe())

# calculating the mean of excess_returns
avg_excess_return = excess_returns.mean()

# plotting avg_excess_returns
avg_excess_return.plot.bar(title='Mean of the Return Difference')
plt.show()

# calculating the standard deviations
sd_excess_return = excess_returns.std()

# plotting the standard deviations
sd_excess_return.plot.bar(title='Standard Deviation of the Return Difference')
plt.show()

# calculating the daily sharpe ratio
daily_sharpe_ratio = avg_excess_return.div(sd_excess_return)

# annualized the sharpe ratio
annual_factor = np.sqrt(252)
annual_sharpe_ratio = daily_sharpe_ratio.mul(annual_factor)

# plotting the annualized sharpe ratio
annual_sharpe_ratio.plot.bar(title='Annualized Sharpe Ratio: Stocks vs S&P 500')
plt.show()

#buy_amazon = True
buy_facebook = True

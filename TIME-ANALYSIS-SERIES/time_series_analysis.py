#!/usr/bin/env python
# coding: utf-8

# In[3]:


# Importing libraries
from plotly import tools

from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff


# In[ ]:





# In[ ]:





# In[4]:


import warnings
warnings.filterwarnings('ignore')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight') 
# Above is a special style template for matplotlib, highly useful for visualizing time series data
get_ipython().run_line_magic('matplotlib', 'inline')
from pylab import rcParams

import statsmodels.api as sm
from numpy.random import normal, seed
from scipy.stats import norm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.tsa.arima_model import ARIMA
import math
from sklearn.metrics import mean_squared_error


# In[5]:


# importing data
# Google Stocks Data,Humidity in different world cities,Microsoft Stocks Data,Pressure in different world cities


# In[6]:


google = pd.read_csv('D:/ml/stock-time-series-20050101-to-20171231/GOOGL_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])
google.head()


# In[7]:


humidity = pd.read_csv('D:/ml/historical-hourly-weather-data/humidity.csv', index_col='datetime', parse_dates=['datetime'])
humidity.tail()


# In[8]:


humidity.describe()


# In[9]:


google.describe()


# In[10]:


google.info()


# In[11]:


humidity.info()


# In[12]:


# Cleaning and preparing time series data
#Google stocks data doesn't have any missing values but humidity data does have its fair share of missing values. It is cleaned using fillna() method with ffill parameter which propagates last valid observation to fill gaps 


# In[13]:


humidity.tail()


# In[14]:


humidity = humidity.iloc[1:]


# In[15]:


humidity = humidity.fillna(method='ffill')


# In[129]:


humidity["Montreal"].diff().iloc[1:].values.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[17]:


# Visualizing the datasets
humidity["Kansas City"].asfreq('M').plot() # asfreq method is used to convert a time series to a specified frequency. Here it is monthly frequency.
plt.title('Humidity in Kansas City over time(Monthly frequency)')
plt.show()


# In[18]:


google['2008':'2010'].plot(subplots=True, figsize=(10,12))
plt.title('Google stock attributes from 2008 to 2010')
plt.savefig('stocks.png')
plt.show()
# if sublot paramater is false no open high low andd close graph will be there


# In[19]:


# Timestamps and Periods
# Timestamps are used to represent a point in time. Periods represent an interval in time. Periods can used to check if a specific event in the given period. They can also be converted to each other's form.


# In[20]:


# Creating a Timestamp
timestamp = pd.Timestamp(2017, 1, 1, 12)
timestamp


# In[21]:


# Creating a period
period = pd.Period('2017-01-01')
period


# In[22]:


# Checking if the given timestamp exists in the given period
period.start_time < timestamp < period.end_time


# In[23]:


# Converting timestamp to period
new_period = timestamp.to_period(freq='H')
new_period


# In[24]:


# Converting period to timestamp
new_timestamp = period.to_timestamp(freq='H', how='start')
new_timestamp


# In[25]:


# date_range is a method that returns a fixed frequency datetimeindex. It is quite useful when creating your own time series attribute for pre-existing data or arranging the whole data around the time series attribute created by you.


# In[26]:


# Creating a datetimeindex with daily frequency
dr1 = pd.date_range(start='1/1/18', end='1/9/18')
dr1


# In[27]:


# Creating a datetimeindex with monthly frequency
dr2 = pd.date_range(start='1/1/18', end='1/1/19', freq='M')
dr2


# In[28]:


# Creating a datetimeindex without specifying start date and using periods
dr3 = pd.date_range(end='1/4/2014', periods=8)
dr3


# In[29]:


# Creating a datetimeindex specifying start date , end date and periods
dr4 = pd.date_range(start='2013-04-24', end='2014-11-27', periods=3)
dr4


# In[30]:


#  Using to_datetime
# pandas.to_datetime() is used for converting arguments to datetime. Here, a DataFrame is converted to a datetime series.


# In[31]:


df = pd.DataFrame({'year': [2015, 2016], 'month': [2, 3], 'day': [4, 5]})
df


# In[32]:


df = pd.to_datetime(df)
df


# In[33]:


df = pd.to_datetime('01-01-2017')
df


# In[35]:


# Shifting and lags
# We can shift index by desired number of periods with an optional time frequency. This is useful when comparing the time series with a past of itself
humidity["Vancouver"].asfreq('M').plot(legend=True)
shifted = humidity["Vancouver"].asfreq('M').shift(10).plot(legend=True)
shifted.legend(['Vancouver','Vancouver_lagged'])
plt.show()


# In[41]:


# Resampling
# Upsampling - Time series is resampled from low frequency to high frequency(Monthly to daily frequency). It involves filling or interpolating missing data
# Downsampling - Time series is resampled from high frequency to low frequency(Weekly to monthly frequency). It involves aggregation of existing data.
# First, we used ffill parameter which propagates last valid observation to fill gaps. Then we use bfill to propogate next valid observation to fill gaps.


# In[37]:


# # Let's use pressure data to demonstrate this
pressure = pd.read_csv('D:/ml/historical-hourly-weather-data/pressure.csv', index_col='datetime', parse_dates=['datetime'])
pressure.tail()


# In[39]:


pressure = pressure.iloc[1:]
pressure = pressure.fillna(method='ffill')
# ffil mean previously filled value along  that axis
pressure.tail()


# In[40]:


# bfil mean next  filled value along  that axis

pressure = pressure.fillna(method='bfill')
pressure.head()


# In[42]:


# Shape before resampling(downsampling)
pressure.shape


# In[43]:


pressure.head()


# In[44]:


# We downsample from hourly to 3 day frequency aggregated using mean
pressure = pressure.resample('3D').mean()
pressure.head()


# In[45]:


# Shape after resampling(downsampling)
pressure.shape


# In[46]:


# Much less rows are left. Now, we will upsample from 3 day frequency to daily frequency


# In[47]:


pressure = pressure.resample('D').pad()
pressure.head()


# In[48]:


# Shape after resampling(upsampling)
pressure.shape


# In[49]:


#  Finance and statistics


# In[55]:


#  Percent change
google['Change'] = google.High.div(google.High.shift())
google['Change'].plot(figsize=(20,8))


# In[57]:


#  Stock returns
google['Return'] = google.Change.sub(1).mul(100)
google['Return'].plot(figsize=(20,8))


# In[58]:


google.High.pct_change().mul(100).plot(figsize=(20,6)) # Another way to calculate returns


# In[59]:


# Absolute change in successive rows
google.High.diff().plot(figsize=(20,6))


# In[60]:


# Comaring two or more time series
# We will compare 2 time series by normalizing them. This is achieved by dividing each time series element of all time series by the first element. This way both series start at the same point and can be easily compared.


# In[61]:


# # We choose microsoft stocks to compare them with google
microsoft = pd.read_csv('D:/ml/stock-time-series-20050101-to-20171231/MSFT_2006-01-01_to_2018-01-01.csv', index_col='Date', parse_dates=['Date'])


# In[62]:


# Plotting before normalization
google.High.plot()
microsoft.High.plot()
plt.legend(['Google','Microsoft'])
plt.show()


# In[63]:


# Normalizing and comparison
# Both stocks start from 100
normalized_google = google.High.div(google.High.iloc[0]).mul(100)
normalized_microsoft = microsoft.High.div(microsoft.High.iloc[0]).mul(100)
normalized_google.plot()
normalized_microsoft.plot()
plt.legend(['Google','Microsoft'])
plt.show()


# In[64]:


# You can clearly see how google outperforms microsoft over time.
# Window functions
# Window functions are used to identify sub periods, calculates sub-metrics of sub-periods.
# Rolling - Same size and sliding
# Expanding - Contains all prior values


# In[65]:


# Rolling window functions
rolling_google = google.High.rolling('90D').mean()
google.High.plot()
rolling_google.plot()
plt.legend(['High','Rolling Mean'])
# Plotting a rolling mean of 90 day window with original High attribute of google stocks
plt.show()


# In[66]:


# Now, observe that rolling mean plot is a smoother version of the original plot.


# In[67]:


# Expanding window functions
microsoft_mean = microsoft.High.expanding().mean()
microsoft_std = microsoft.High.expanding().std()
microsoft.High.plot()
microsoft_mean.plot()
microsoft_std.plot()
plt.legend(['High','Expanding Mean','Expanding Standard Deviation'])
plt.show()


# In[68]:


# OHLC charts


# In[69]:


"""
An OHLC chart is any type of price chart that shows the open, high, low and close price of a certain time period. Open-high-low-close Charts (or OHLC Charts) are used as a trading tool to visualise and analyse the price changes over time for securities, currencies, stocks, bonds, commodities, etc. OHLC Charts are useful for interpreting the day-to-day sentiment of the market and forecasting any future price changes through the patterns produced.

The y-axis on an OHLC Chart is used for the price scale, while the x-axis is the timescale. On each single time period, an OHLC Charts plots a symbol that represents two ranges: the highest and lowest prices traded, and also the opening and closing price on that single time period (for example in a day). On the range symbol, the high and low price ranges are represented by the length of the main vertical line. The open and close prices are represented by the vertical positioning of tick-marks that appear on the left (representing the open price) and on right (representing the close price) sides of the high-low vertical line.
Colour can be assigned to each OHLC Chart symbol, to distinguish whether the market is "bullish" (the closing price is higher then it opened) or "bearish" (the closing price is lower then it opened).
"""


# In[70]:


# OHLC chart of June 2008
trace = go.Ohlc(x=google['06-2008'].index,
                open=google['06-2008'].Open,
                high=google['06-2008'].High,
                low=google['06-2008'].Low,
                close=google['06-2008'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')


# In[71]:


# OHLC chart of 2008
trace = go.Ohlc(x=google['2008'].index,
                open=google['2008'].Open,
                high=google['2008'].High,
                low=google['2008'].Low,
                close=google['2008'].Close)
data = [trace]
iplot(data, filename='simple_ohlc')


# In[72]:


#  Candlestick charts
"""This type of chart is used as a trading tool to visualise and analyse the price movements over time for securities, derivatives, currencies, stocks, bonds, commodities, etc. Although the symbols used in Candlestick Charts resemble a Box Plot, they function differently and therefore, are not to be confused with one another.

Candlestick Charts display multiple bits of price information such as the open price, close price, highest price and lowest price through the use of candlestick-like symbols. Each symbol represents the compressed trading activity for a single time period (a minute, hour, day, month, etc). Each Candlestick symbol is plotted along a time scale on the x-axis, to show the trading activity over time.

The main rectangle in the symbol is known as the real body, which is used to display the range between the open and close price of that time period. While the lines extending from the bottom and top of the real body is known as the lower and upper shadows (or wick). Each shadow represents the highest or lowest price traded during the time period represented. When the market is Bullish (the closing price is higher than it opened), then the body is coloured typically white or green. But when the market is Bearish (the closing price is lower than it opened), then the body is usually coloured either black or red.
Candlestick Charts are great for detecting and predicting market trends over time and are useful for interpreting the day-to-day sentiment of the market, through each candlestick symbol's colouring and shape. For example, the longer the body is, the more intense the selling or buying pressure is. While, a very short body, would indicate that there is very little price movement in that time period and represents consolidation.

Candlestick Charts help reveal the market psychology (the fear and greed experienced by sellers and buyers) through the various indicators, such as shape and colour, but also by the many identifiable patterns that can be found in Candlestick Charts. In total, there are 42 recognised patterns that are divided into simple and complex patterns. These patterns found in Candlestick Charts are useful for displaying price relationships and can be used for predicting the possible future movement of the market. You can find a list and description of each pattern here.

Please bear in mind, that Candlestick Charts don't express the events taking place between the open and close price - only the relationship between the two prices. So you can't tell how volatile trading was within that single time period.
"""


# In[73]:


# Candlestick chart of march 2008
trace = go.Candlestick(x=google['03-2008'].index,
                open=google['03-2008'].Open,
                high=google['03-2008'].High,
                low=google['03-2008'].Low,
                close=google['03-2008'].Close)
data = [trace]
iplot(data, filename='simple_candlestick')


# In[74]:


# Candlestick chart of 2008
trace = go.Candlestick(x=google['2008'].index,
                open=google['2008'].Open,
                high=google['2008'].High,
                low=google['2008'].Low,
                close=google['2008'].Close)
data = [trace]
iplot(data, filename='simple_candlestick')


# In[75]:


# Candlestick chart of 2006-2018
trace = go.Candlestick(x=google.index,
                open=google.Open,
                high=google.High,
                low=google.Low,
                close=google.Close)
data = [trace]
iplot(data, filename='simple_candlestick')


# In[ ]:


# Autocorrelation and Partial Autocorrelation¶
# Autocorrelation - The autocorrelation function (ACF) measures how a series is correlated with itself at different lags.
# Partial Autocorrelation - The partial autocorrelation function can be interpreted as a regression of the series against its past lags. The terms can be interpreted the same way as a standard linear regression, that is the contribution of a change in that particular lag while holding others constant.


# In[76]:


# Autocorrelation of humidity of San Diego
plot_acf(humidity["San Diego"],lags=25,title="San Diego")
plt.show()


# In[77]:


# As all lags are either close to 1 or at least greater than the confidence interval, they are statistically significant.


# In[78]:


# Partial Autocorrelation of humidity of San Diego
plot_pacf(humidity["San Diego"],lags=25)
plt.show()


# In[79]:


# Though it is statistically signficant, partial autocorrelation after first 2 lags is very low.


# In[80]:


# Partial Autocorrelation of closing price of microsoft stocks
plot_pacf(microsoft["Close"],lags=25)
plt.show()


# In[81]:


# Here, only 0th, 1st and 20th lag are statistically significan


# In[82]:


# Time series decomposition and Random walks


# In[ ]:


# Trends, seasonality and noise
#     These are the components of a time series
#     Trend - Consistent upwards or downwards slope of a time series
#     Seasonality - Clear periodic pattern of a time series(like sine funtion)
#     Noise - Outliers or missing values


# In[83]:


# Let's take Google stocks High for this
google["High"].plot(figsize=(16,8))


# In[84]:


# Now, for decomposition...
rcParams['figure.figsize'] = 11, 9
decomposed_google_volume = sm.tsa.seasonal_decompose(google["High"],freq=360) # The frequncy is annual
figure = decomposed_google_volume.plot()
plt.show()


# In[ ]:


# There is clearly an upward trend in the above plot.
# You can also see the uniform seasonal change.
# Non-uniform noise that represent outliers and missing values


# In[85]:


# White noise
# White noise has...
# Constant mean
# Constant variance
# Zero auto-correlation at all lags


# In[86]:


# Plotting white noise
rcParams['figure.figsize'] = 16, 6
white_noise = np.random.normal(loc=0, scale=1, size=1000)
# loc is mean, scale is variance
plt.plot(white_noise)


# In[87]:


# Plotting autocorrelation of white noise
plot_acf(white_noise,lags=20)
plt.show()


# In[ ]:


# See how all lags are statistically insigficant as they lie inside the confidence interval(shaded portion).


# In[ ]:


#  Random Walk
"""
A random walk is a mathematical object, known as a stochastic or random process, that describes a path that consists of a succession of random steps on some mathematical space such as the integers.

In general if we talk about stocks, Today's Price = Yesterday's Price + Noise

Pt = Pt-1 + εt
Random walks can't be forecasted because well, noise is random.

Random Walk with Drift(drift(μ) is zero-mean)

Pt - Pt-1 = μ + εt

Regression test for random walk

Pt = α + βPt-1 + εt
Equivalent to Pt - Pt-1 = α + βPt-1 + εt

Test:

H0: β = 1 (This is a random walk)
H1: β < 1 (This is not a random walk)

Dickey-Fuller Test:

H0: β = 0 (This is a random walk)
H1: β < 0 (This is not a random walk)
Augmented Dickey-Fuller test
An augmented Dickey–Fuller test (ADF) tests the null hypothesis that a unit root is present in a time series sample. It is basically Dickey-Fuller test with more lagged changes on RHS


# In[88]:


# Augmented Dickey-Fuller test on volume of google and microsoft stocks 
adf = adfuller(microsoft["Volume"])
print("p-value of microsoft: {}".format(float(adf[1])))
adf = adfuller(google["Volume"])
print("p-value of google: {}".format(float(adf[1])))


# In[ ]:


# both are rejected as p value less than 0.05 so null ypothesis is rejected and this is not a random walk.


# In[ ]:


# Generating a random walk


# In[89]:


seed(42)
rcParams['figure.figsize'] = 16, 6
random_walk = normal(loc=0, scale=0.01, size=1000)
plt.plot(random_walk)
plt.show()


# In[90]:


fig = ff.create_distplot([random_walk],['Random Walk'],bin_size=0.001)
iplot(fig, filename='Basic Distplot')


# In[ ]:


"""
Stationarity
A stationary time series is one whose statistical properties such as mean, variance, autocorrelation, etc. are all constant over time.

Strong stationarity: is a stochastic process whose unconditional joint probability distribution does not change when shifted in time. Consequently, parameters such as mean and variance also do not change over time.
Weak stationarity: is a process where mean, variance, autocorrelation are constant throughout the time

Stationarity is important as non-stationary series that depend on time have too many parameters to account for when modelling the time series. diff() method can easily convert a non-stationary series to a stationary series.

We will try to decompose seasonal component of the above decomposed time series.
"""


# In[94]:


# The original non-stationary plot
decomposed_google_volume.trend.plot()


# In[92]:


# The new stationary plot
decomposed_google_volume.trend.diff().plot()


# In[95]:


# Modelling using statstools¶


# In[96]:


#  AR modelS
# An autoregressive (AR) model is a representation of a type of random process; as such, it is used to describe certain time-varying processes in nature, economics, etc. The autoregressive model specifies that the output variable depends linearly on its own previous values and on a stochastic term (an imperfectly predictable term); thus the model is in the form of a stochastic difference equation.

# AR(1) model
# Rt = μ + ϕRt-1 + εt

# As RHS has only one lagged value(Rt-1)this is called AR model of order 1 where μ is mean and ε is noise at time t
# If ϕ = 1, it is random walk. Else if ϕ = 0, it is white noise. Else if -1 < ϕ < 1, it is stationary. If ϕ is -ve, there is men reversion. If ϕ is +ve, there is momentum.

# AR(2) model
# Rt = μ + ϕ1Rt-1 + ϕ2Rt-2 + εt

# AR(3) model
# Rt = μ + ϕ1Rt-1 + ϕ2Rt-2 + ϕ3Rt-3 + εt


# In[97]:


# Simulating AR(1) model
rcParams['figure.figsize'] = 16, 12
plt.subplot(4,1,1)
ar1 = np.array([1, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma1 = np.array([1])
AR1 = ArmaProcess(ar1, ma1)
sim1 = AR1.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = +0.9')
plt.plot(sim1)
# We will take care of MA model later
# AR(1) MA(1) AR parameter = -0.9
plt.subplot(4,1,2)
ar2 = np.array([1, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma2 = np.array([1])
AR2 = ArmaProcess(ar2, ma2)
sim2 = AR2.generate_sample(nsample=1000)
plt.title('AR(1) model: AR parameter = -0.9')
plt.plot(sim2)
# AR(2) MA(1) AR parameter = 0.9
plt.subplot(4,1,3)
ar3 = np.array([2, -0.9]) # We choose -0.9 as AR parameter is +0.9
ma3 = np.array([1])
AR3 = ArmaProcess(ar3, ma3)
sim3 = AR3.generate_sample(nsample=1000)
plt.title('AR(2) model: AR parameter = +0.9')
plt.plot(sim3)
# AR(2) MA(1) AR parameter = -0.9
plt.subplot(4,1,4)
ar4 = np.array([2, 0.9]) # We choose +0.9 as AR parameter is -0.9
ma4 = np.array([1])
AR4 = ArmaProcess(ar4, ma4)
sim4 = AR4.generate_sample(nsample=1000)
plt.title('AR(2) model: AR parameter = -0.9')
plt.plot(sim4)
plt.show()


# In[98]:


# Forecasting a simulated model
model = ARMA(sim1, order=(1,0))
result = model.fit()
print(result.summary())
print("μ={} ,ϕ={}".format(result.params[0],result.params[1]))


# In[100]:


# ϕ is around 0.9 which is what we chose as AR parameter in our first simulated model.


# In[101]:


# Predicting the models


# In[102]:


# Predicting simulated AR(1) model 
result.plot_predict(start=900, end=1010)
plt.show()


# In[103]:


rmse = math.sqrt(mean_squared_error(sim1[900:1011], result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))


# In[122]:


# Predicting humidity level of Montreal
humid = ARMA(humidity["Montreal"].diff().iloc[1:].values, order=(1,0))
res = humid.fit()
res.plot_predict(start=1000, end=1100)
plt.show()
res.predict(start=27,end=36)


# In[ ]:





# In[ ]:





# In[ ]:





# In[105]:


rmse = math.sqrt(mean_squared_error(humidity["Montreal"].diff().iloc[900:1000].values, result.predict(start=900,end=999)))
print("The root mean squared error is {}.".format(rmse))


# In[106]:


# Predicting closing prices of google
humid = ARMA(google["Close"].diff().iloc[1:].values, order=(1,0))
res = humid.fit()
res.plot_predict(start=900, end=1010)
plt.show()


# In[107]:


# The moving-average (MA) model is a common approach for modeling univariate time series. The moving-average model specifies that the output variable depends linearly on the current and various past values of a stochastic (imperfectly predictable) term.

# MA(1) model
# Rt = μ + ϵt1 + θϵt-1

# It translates to Today's returns = mean + today's noise + yesterday's noise

# As there is only 1 lagged value in RHS, it is an MA model of order 1


# In[108]:


rcParams['figure.figsize'] = 16, 6
ar1 = np.array([1])
ma1 = np.array([1, -0.5])
MA1 = ArmaProcess(ar1, ma1)
sim1 = MA1.generate_sample(nsample=1000)
plt.plot(sim1)


# In[109]:


# Forecasting the simulated MA model


# In[110]:


model = ARMA(sim1, order=(0,1))
result = model.fit()
print(result.summary())
print("μ={} ,θ={}".format(result.params[0],result.params[1]))


# In[172]:


####  Prediction using MA models¶


# In[ ]:





# In[112]:


# Forecasting and predicting montreal humidity
model = ARMA(humidity["Montreal"].diff().iloc[1:].values, order=(0,3))
result = model.fit()
print(result.summary())
print("μ={} ,θ={}".format(result.params[0],result.params[1]))
result.plot_predict(start=1000, end=1100)
plt.show()


# In[113]:


rmse = math.sqrt(mean_squared_error(humidity["Montreal"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))
print("The root mean squared error is {}.".format(rmse))


# In[ ]:


# ARMA models
# Autoregressive–moving-average (ARMA) models provide a parsimonious description of a (weakly) stationary stochastic process in terms of two polynomials, one for the autoregression and the second for the moving average. It's the fusion of AR and MA models.

# ARMA(1,1) model
# Rt = μ + ϕRt-1 + ϵt + θϵt-1
# Basically, Today's return = mean + Yesterday's return + noise + yesterday's noise.


# In[ ]:


# Prediction using ARMA models
# I am not simulating any model because it's quite similar to AR and MA models. Just forecasting and predictions for this one.


# In[115]:


# Forecasting and predicting microsoft stocks volume
model = ARMA(microsoft["Volume"].diff().iloc[1:].values, order=(3,3))
result = model.fit()
print(result.summary())
print("μ={}, ϕ={}, θ={}".format(result.params[0],result.params[1],result.params[2]))
result.plot_predict(start=1000, end=1100)
plt.show()


# In[117]:


rmse = math.sqrt(mean_squared_error(microsoft["Volume"].diff().iloc[1000:1101].values, result.predict(start=1000,end=1100)))
print("The root mean squared error is {}.".format(rmse))


# In[ ]:


"""
 ARIMA models
An autoregressive integrated moving average (ARIMA) model is a generalization of an autoregressive moving average (ARMA) model. Both of these models are fitted to time series data either to better understand the data or to predict future points in the series (forecasting). ARIMA models are applied in some cases where data show evidence of non-stationarity, where an initial differencing step (corresponding to the "integrated" part of the model) can be applied one or more times to eliminate the non-stationarity. ARIMA model is of the form: ARIMA(p,d,q): p is AR parameter, d is differential parameter, q is MA parameter

ARIMA(1,0,0)
yt = a1yt-1 + ϵt

ARIMA(1,0,1)
yt = a1yt-1 + ϵt + b1ϵt-1

ARIMA(1,1,1)
Δyt = a1Δyt-1 + ϵt + b1ϵt-1 where Δyt = yt - yt-1

"""


# In[118]:


# Predicting the microsoft stocks volume
rcParams['figure.figsize'] = 16, 6
model = ARIMA(microsoft["Volume"].diff().iloc[1:].values, order=(2,1,0))
result = model.fit()
print(result.summary())
result.plot_predict(start=700, end=1000)
plt.show()


# In[135]:


rmse = math.sqrt(mean_squared_error(microsoft["Volume"].diff().iloc[700:1001].values, result.predict(start=700,end=1000)))
print("The root mean squared error is {}.".format(rmse))


# In[168]:


result.predict(start=700, end=1000)[240]


# In[171]:


microsoft["Volume"].diff().iloc[1:][940]


# In[ ]:


"""

1. Introduction to date and time
    1.1 Importing time series data
    1.2 Cleaning and preparing time series data
    1.3 Visualizing the datasets
    1.4 Timestamps and Periods
    1.5 Using date_range
    1.6 Using to_datetime
    1.7 Shifting and lags
    1.8 Resampling
2. Finance and Statistics
    2.1 Percent change
    2.2 Stock returns
    2.3 Absolute change in successive rows
    2.4 Comaring two or more time series
    2.5 Window functions
    2.6 OHLC charts
    2.7 Candlestick charts
    2.8 Autocorrelation and Partial Autocorrelation
3. Time series decomposition and Random Walks
    3.1 Trends, Seasonality and Noise
    3.2 White Noise
    3.3 Random Walk
    3.4 Stationarity
4. Modelling using statsmodels
    4.1 AR models
    4.2 MA models
    4.3 ARMA models
    4.4 ARIMA models
"""


# In[ ]:





# In[ ]:





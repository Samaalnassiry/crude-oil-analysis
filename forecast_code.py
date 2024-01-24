import pandas as pd #data processing
import pastas as ps #alt option
import matplotlib.pyplot as plt #plotting
import numpy as np #linear algebra
from sklearn.metrics import mean_squared_error as rmse
from sklearn.metrics import mean_absolute_percentage_error as mape
from sktime.performance_metrics.forecasting import mean_absolute_scaled_error as mase
import statsmodels.api as sm
import seaborn as sns #plotting
import statsmodels.graphics.tsaplots as sgt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
import pymannkendall as mk #Mann-Kendall trend test
from statsmodels.tsa.stattools import adfuller #Augumented Dickey-Fuller
from prophet import Prophet #time series predition library
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric
from IPython.display import display
pd.options.mode.chained_assignment = None  # default='warn'
from warnings import filterwarnings
filterwarnings('ignore')





pd.options.display.max_seq_items = None
path_to_data = '/Users/samaalnassiry/Desktop/Crude Oil Analysis/clean-wide-final.csv'
open_data = pd.read_csv(path_to_data)
open_data.head() #reading first few rows


def country_selector(x): #parses through the dataframe to return only data from one country
  internal_country = open_data.loc[open_data['_LOCATION_']==x]
  time_val = internal_country[['TIME','Value']].sort_values(by='TIME', ascending=True) #pulls time 
  #and production value
 #returns by time ascending
  roc_time_val = time_val.Value.pct_change() #calculates the rate of change of production
  time_val.insert(2,'Change',roc_time_val,True) #appends the ROC
  return time_val

country = country_selector('United States')
print(country)


#Time vs production line plot
plt.figure(figsize=(8,4))
sns.lineplot(data=country, x = 'TIME' , y = 'Value')
plt.xlabel('Year', fontsize='16', horizontalalignment='center')
plt.ylabel('KTOE',fontsize='16', horizontalalignment='center')
plt.title('Yearly Oil Production Value')
plt.show()

#Time vs production change line plot
plt.figure(figsize=(8,4))
sns.lineplot(data=country, x = 'TIME' , y = 'Change')
plt.xlabel('Year', fontsize='16', horizontalalignment='center')
plt.ylabel('Change',fontsize='16', horizontalalignment='center')
plt.title('Change in Production Value')
plt.show()


country = country.rename(columns = {'TIME':'ds','Value':'y'}) #converting values to fit prophet
#Time series model
model = Prophet()
model.fit(country[['ds','y']])

#Future dataframe
future = model.make_future_dataframe(periods=60,freq='Y')

forecast = model.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(12)
forecast.drop(forecast.index[:50],inplace=True)\

fig1 = model.plot(forecast, xlabel='Year', ylabel='KTOE',
    figsize=(8, 4))



#Mann-Kendall Test on present day data, H1 = a trend is avaliable in the series
#H0 = No trend is avaliable

print(mk.original_test(country.y))

#Augmented Dickey-Fuller test for stationarity
adftest= adfuller(country['y'], autolag='AIC', regression='ct')
print("ADF-Statistic:", adftest[0])
print("P-Value:", adftest[1])
print("Number of lags:", adftest[2])
print("Number of observations:", adftest[3])
print("Critical Values:", adftest[4])
print("Note: If P-Value is smaller than 0.05, we reject the null hypothesis and the series is stationary")

#Spearman's correlation
def display_correlation(df):
    r = df.corr(method="spearman")
    plt.figure(figsize=(8,5))
    heatmap = sns.heatmap(df.corr(), vmin=-1, 
                      vmax=1, annot=True)
    plt.title("Spearman Correlation")
    return(r)
spr = display_correlation(country)


#ACF
fig, ax = plt.subplots(1,2, figsize=(10,5))
sm.graphics.tsa.plot_acf(country['y'], lags=20, ax=ax[0])
#PACF
sm.graphics.tsa.plot_pacf(country['y'], lags=9, ax=ax[1])
plt.show()

#Preping for exponential smoothing
#30% of the data will be held for training
df = pd.DataFrame(country[['ds','y']])
df = df.rename(columns = {'ds':'Year','y':'KTOE'})

#Sets the frequency of the enteries, turns years into datetime and index
df['Year'] = pd.date_range(start='2001-12-31', periods=21, freq='A-DEC')
df.set_index('Year', inplace = True)


dfx = df #copy of full original df
test = dfx.tail(6) #holdout data
df = dfx[:-6] #training data
pred = test.copy()
display(df,test)

#Simple Exponential Smoothing
ses = SimpleExpSmoothing(df['KTOE'])
alpha = 0.8
smooth = 0.8
model = ses.fit(smoothing_level = alpha, optimized=False)
df['SES'] = model.fittedvalues
stationaryforecast = pd.DataFrame(model.forecast(6))
df.head()

# Double exponential smoothing
model = ExponentialSmoothing(df['KTOE'],trend= 'add',freq = 'A-DEC')
des_model = model.fit(smoothing_level=alpha, smoothing_trend=smooth)
df2 =df
df2['DES']=des_model.fittedvalues
df2.head()

# Perform triple exponential smoothing
model = ExponentialSmoothing(df2['KTOE'], trend='mul',freq = 'A-DEC', seasonal='mul', seasonal_periods=2)
tes_model = model.fit(smoothing_level=alpha, smoothing_trend=smooth)
model = model.fit()
df3 = df2
df3['TES'] = tes_model.fittedvalues
df3.head()

#Exponential forecast
d = pd.DataFrame(model.forecast(6))
forecast = pd.DataFrame(data=d)
forecast.index.names = ['Year']
forecast = forecast.rename(columns={forecast.columns[0]:'Forecast'})

forecast.head()

#ARIMA
train_arima = df['KTOE']
test_arima = test['KTOE']

history = [x for x in train_arima]
y = test_arima
predictions = []
# rolling forecasts
for i in range(0, len(y)):
    # predict
    model = ARIMA(history, order=(1,2,2))
    model_fit = model.fit()
    yhat = model_fit.forecast()[0]
    # invert transformed prediction
    predictions.append(yhat)
    # observation
    obs = y[i]
    history.append(obs)
test['ARIMA'] = predictions

#Exponential Smoothing Plot
plt.figure(figsize=(8, 5))
plt.plot(dfx['KTOE'], label='Actual Production', marker='o')
plt.plot(df['SES'], label='SES')
plt.plot(df2['DES'], label='DES')
plt.plot(df3['TES'], label='TES')
plt.xlabel('Year')
plt.ylabel('Production in KTOE')
plt.title('Exponential Smoothing')
plt.legend()
plt.show()

#Forecasting plot
plt.figure(figsize=(8, 5))
plt.plot(dfx['KTOE'], label='Actual Production', marker='o')
plt.plot(stationaryforecast, label='Stationary Forecast')
plt.plot(test['ARIMA'], color='springgreen',label='ARIMA Predictions')
plt.plot(forecast, label='Exponential Forecast')
plt.xlabel('Year')
plt.ylabel('Production in KTOE')
plt.title('Forecasting')
plt.legend()
plt.show()

#Visualize actual vs predicted values of forecast on training sets
y_t =  df3['KTOE']
y_ses = df3['SES']
y_des = df3['DES']
y_tes = df3['TES']
y_arm = test['ARIMA']
fig, ax = plt.subplots()
ax.scatter(range(len(y_t)), y_t, color='blue', label='Actual')
ax.scatter(range(len(y_ses)), y_ses, color='orange', label='SES')
ax.scatter(range(len(y_des)), y_des, color='green', label='DES')
ax.scatter(range(len(y_tes)), y_tes, color='red', label='TES')
ax.set_xlabel('Index')
ax.set_ylabel('Target Value')
ax.set_title('Actual vs. Predicted Target Values on Test Set')
ax.legend()
plt.show()

# Visualize the actual vs predicted values of the forecast on test set
y_true = test['KTOE']
y_pred = forecast['Forecast'] 
fig, ax = plt.subplots()
ax.scatter(range(len(y_true)), y_true, color='blue', label='Actual')
ax.scatter(range(len(y_pred)), y_pred, color='brown', label='Exponential Forecast')
ax.scatter(range(len(y_arm)), y_arm, color='springgreen', label='ARIMA')
ax.set_xlabel('Index')
ax.set_ylabel('Target Value')
ax.set_title('Actual vs. Predicted Target Values on Test Set')
ax.legend()
plt.show()

#Compute accuracy of models
def getAccuracy(x,y):
    rmse_p = rmse(x,y,squared = False)
    mape_p = mape(x,y)*100
    mase_p = mase(x,y,y_train=y)
    data = {'Property':['Root Mean Squared Error','Mean Absolute Scaled Error','Mean Absolute Percentage Error'],
        'Value':[rmse_p,mase_p,mape_p]
        }
    ds= pd.DataFrame(data)
    ds = ds.set_index('Property')
    return ds

SES = getAccuracy(y_ses,y_t)
DES = getAccuracy(y_des,y_t)
TES = getAccuracy(y_tes,y_t)
forc = getAccuracy(y_pred,y_true)
arim = getAccuracy(y_arm, y_true)
display(SES.style.set_caption('SES Accuracy'),
        DES.style.set_caption('DES Accuracy'),
        TES.style.set_caption('TES Accuracy'),
        forc.style.set_caption('Exponential Forecast Accuracy'),
        arim.style.set_caption('ARIMA Forecast Accuracy'))

#Predictive models accuracy analysis without 2020
forc20 = getAccuracy(y_true.iloc[:-2],y_pred.iloc[:-2])
arim20 = getAccuracy(y_arm.iloc[:-2], y_true.iloc[:-2])

display(forc20.style.set_caption('Exponential Forecast Accuracy Adjusted to 2020'))
display(arim20.style.set_caption('ARIMA Forecast Accuracy Adjusted to 2020'))
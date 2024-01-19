#importing analysis libraries
import pandas as pd #data processing
import matplotlib.pyplot as plt #plotting
import numpy as np #linear algebra
import seaborn as sns #plotting
from prophet import Prophet #time series predition library
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics
from prophet.plot import plot_cross_validation_metric




pd.options.display.max_seq_items = None
path_to_data = '/Users/samaalnassiry/Desktop/Crude Oil Analysis/clean-wide-final.csv'
open_data = pd.read_csv(path_to_data)
open_data.head() #reading first few rows


def country_selector(x): #function that parses through the dataframe to return only data from one country
  internal_country = open_data.loc[open_data['_LOCATION_']==x]
  time_val = internal_country[['TIME','Value']].sort_values(by='TIME', ascending=True) #pulls time 
  #and production value
 #returns by time ascending
  roc_time_val = time_val.Value.pct_change() #calculates the rate of change of production
  time_val.insert(2,'Change',roc_time_val,True) #appends the ROC
  return time_val

country = country_selector('Iraq')
print(country)

#Time vs production line plot
plt.figure(figsize=(10,8))
sns.lineplot(data=country, x = 'TIME' , y = 'Value')
plt.xlabel('Year', fontsize='17', horizontalalignment='center')
plt.ylabel('KTOE',fontsize='17', horizontalalignment='center')
plt.title('Yearly Oil Production Value')
plt.show()

#Time vs production change line plot
plt.figure(figsize=(10,8))
sns.lineplot(data=country, x = 'TIME' , y = 'Change')
plt.xlabel('Year', fontsize='17', horizontalalignment='center')
plt.ylabel('Change',fontsize='17', horizontalalignment='center')
plt.title('Change in Production Value')
plt.show()


country = country.rename(columns = {'TIME':'ds','Value':'y'}) #converting values to fit prophet
print(country.head())
#Time series model
model = Prophet()
model.fit(country[['ds','y']])

#Future dataframe
future = model.make_future_dataframe(periods=60,freq='Y')

forecast = model.predict(future)
forecast[['ds','yhat','yhat_lower','yhat_upper']].tail(12)
forecast.drop(forecast.index[:52],inplace=True)


fig1 = model.plot(forecast)


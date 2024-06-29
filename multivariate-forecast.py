import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.api import VAR
import numpy as np


data = pd.read_csv("C:\\Users\\aquamanu\\Documents\\FYP\\data\\Thames_Initiative_2009-2017.csv")

data.head()
data.columns
data.info()
data.keys()


df = pd.DataFrame(data, columns=[ 'Gran alkalinity u eq/L', 'Total phosphorus (ug/L)',
                                  'Dissolved chloride (mg Cl/L)', 'Dissolved nitrate (NO3)',
                                  'Dissolved ammonium (NH4) (mg/l)','Dissolved nitrite (mg NO2/L)'])


num_records = df.shape[0]
print("Total records in df:", num_records)

df_clean = df.copy()


duplicate_count = df_clean.duplicated().sum()
print(f"Number of duplicate rows in the DataFrame: {duplicate_count}")

df_clean.isna().sum()
print(df_clean.info())

df_clean['Gran alkalinity u eq/L'] = pd.to_numeric(df_clean['Gran alkalinity u eq/L'], errors='coerce')
df_clean['Total phosphorus (ug/L)'] = pd.to_numeric(df_clean['Total phosphorus (ug/L)'], errors='coerce')
df_clean['Dissolved nitrate (NO3)'] = pd.to_numeric(df_clean['Dissolved nitrate (NO3)'], errors='coerce')
df_clean['Dissolved ammonium (NH4) (mg/l)'] = pd.to_numeric(df_clean['Dissolved ammonium (NH4) (mg/l)'], errors='coerce')
df_clean['Dissolved nitrite (mg NO2/L)'] = pd.to_numeric(df_clean['Dissolved nitrite (mg NO2/L)'], errors='coerce')

df_clean.isna().sum()
df_clean = df_clean.dropna()
df_clean.info()

"""***PRE-PROCESSING COMPLETE***"""

import pandas as pd

forecast = pd.DataFrame()


forecast['Gran alkalinity u eq/L'] = pd.to_numeric(df_clean['Gran alkalinity u eq/L'], errors='coerce')
forecast['Total phosphorus (ug/L)'] = pd.to_numeric(df_clean['Total phosphorus (ug/L)'], errors='coerce')
forecast['Dissolved nitrate (NO3)'] = pd.to_numeric(df_clean['Dissolved nitrate (NO3)'], errors='coerce')
forecast['Dissolved ammonium (NH4) (mg/l)'] = pd.to_numeric(df_clean['Dissolved ammonium (NH4) (mg/l)'], errors='coerce')
forecast['Dissolved nitrite (mg NO2/L)'] = pd.to_numeric(df_clean['Dissolved nitrite (mg NO2/L)'], errors='coerce')
forecast['Dissolved chloride (mg Cl/L)'] = pd.to_numeric(df_clean['Dissolved chloride (mg Cl/L)'], errors='coerce')

forecast['Sampling Date'] = data['Sampling Date']


print(forecast.head())


feature_counts = forecast.count()


print(feature_counts)


"""***Forecasting Process Begins***"""

forecast.set_index('Sampling Date', inplace=True)
print(forecast.head())

print(forecast.isnull().sum())


forecast.reset_index(drop=True, inplace=True)


forecast.index = pd.to_datetime(forecast.index, unit='D', origin='2009-01-01')


forecast_weekly = forecast.resample('W').mean()
print(forecast_weekly.head())





train_data = forecast_weekly.loc[:'2016-12-12']
test_data = forecast_weekly.loc['2017-01-02':]

print("Training data:")
print(train_data.tail())

print("\nTesting data:")
print(test_data.head())

print("----------------------WWWWWWWWWWWWWWWWWWWWWWWWWWW-------------------------------------")
# adapated from https://www.statsmodels.org/dev/vector_ar.html#


var_features = ['Gran alkalinity u eq/L', 'Total phosphorus (ug/L)', 'Dissolved nitrate (NO3)',
                'Dissolved ammonium (NH4) (mg/l)', 'Dissolved nitrite (mg NO2/L)','Dissolved chloride (mg Cl/L)']
var_data = train_data[var_features]

#  train  VAR model
model = VAR(var_data)
model_fit = model.fit(maxlags=3)


print(model_fit.summary())

#weeks for forecasting
forecast_steps = 52

#fot model
lag_order = model_fit.k_ar
var_forecast = model_fit.forecast(var_data.values[-lag_order:], steps=forecast_steps)

# Convert the forecasted values to df
var_forecast_df = pd.DataFrame(var_forecast, columns=var_features, index=pd.date_range(start=test_data.index[0], periods=forecast_steps, freq='W'))

print("Columns in var_forecast_df:")
print(var_forecast_df.columns)

print("Contents of var_forecast_df:")
print(var_forecast_df.head())

print("Forecasted values:")
print(var_forecast_df[['Gran alkalinity u eq/L', 'Total phosphorus (ug/L)', 'Dissolved nitrate (NO3)',
                       'Dissolved ammonium (NH4) (mg/l)', 'Dissolved nitrite (mg NO2/L)','Dissolved chloride (mg Cl/L)']])


var_forecast_df.to_csv("multivariate_forecasted_values.csv", index=True)
print("Forecasted values saved to multivariate_values.csv")


# Evaluate the model using appropriate metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error
import numpy as np

# Iterate over each feature and calculate evaluation metrics
for feature in var_features:
    print(f"VAR MULTIVARIATE Evaluation metrics for {feature}:")
    
    mse = mean_squared_error(test_data[feature][:forecast_steps], var_forecast_df[feature])
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(test_data[feature][:forecast_steps], var_forecast_df[feature])
    mape = mean_absolute_percentage_error(test_data[feature][:forecast_steps], var_forecast_df[feature])
    
    print("Mean Squared Error (MSE):", mse)
    print("Root Mean Squared Error (RMSE):", rmse)
    print("Mean Absolute Error (MAE):", mae)
    print("Mean Absolute Percentage Error (MAPE):", mape)
    print("---")



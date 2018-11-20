# -*- coding: utf-8 -*-
"""
Created on Fri Nov 16 14:02:30 2018

@author: kjohn
"""

##### This is for ARIMA

import pandas as pd
import numpy as np
from pyramid.arima import auto_arima
import matplotlib.pyplot as plt

data1 = pd.read_csv("C:\\Users\\kjohn\\Desktop\\RPA\\ZScore\\Utility_Electric 2014 -2016.csv")
data1.head()

data1 = data1.iloc[:, 1:37]
no_of_column=len(data1.columns)

forecasted=pd.DataFrame()

for row in data1.iloc[0:20,:].iterrows():
    index, data = row
    if(np.sum(data[no_of_column-3:no_of_column])==0):
        print("Next")
        continue

    data=data.replace(0,np.median(data))
    stepwise_model = auto_arima(data, start_p=1, start_q=1,
                           max_p=3, max_q=3, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
    print(index)
    print(row)
    stepwise_model.aic()
    future_forecast = stepwise_model.predict(n_periods=12)
    forecasted=forecasted.append(pd.Series(future_forecast),ignore_index=True)
    
    # Example
    plt.plot(data)
    plt.plot(future_forecast)
    print(future_forecast)
    
    




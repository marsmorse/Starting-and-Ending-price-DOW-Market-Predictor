import os
import numpy as np
import pandas as pd
from data2csv import save_csv
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from av_key import av_key
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import data2csv
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dropout
from keras.layers import Dense 

# API key
api_key = av_key

# Prompt user for stock symbol
stock = input('Stock: ')
stock = stock.upper()

# Prompt user for time interval (for the data set)
time_interval = input('Daily or Intradaily?: ')
time_interval = time_interval.upper()
print('')

# Prompt user for Indicators 
# Example Input: RSI SMA MACD
indicators = input('Indicators [RSI, SMA, MACD] (Separate by a Space): ')
indicators = indicators.split()

# saves specified dataset to CSV form in /charts
save_csv(stock, time_interval, indicators)
df_price = pd.read_csv(f'./charts/{stock}_{time_interval.lower()}.csv')

# initializes list to [0...0] first
df_ind = [0 for i in range(len(indicators))]
for i in range(len(indicators)):
    # instantiate list with indicator values
    df_ind[i] = pd.read_csv(f'./charts/{stock}_{time_interval.lower()}_{indicators[i]}.csv')

# Create Train and Test data
# For Closing Price
# closing_price_matrix = df_price['4. close'].as_matrix()
# opening_price_matrix = df_price['1. open'].as_matrix()
# high_price_matrix = df_price['2. high'].as_matrix()
# low_price_matrix = df_price['3. low'].as_matrix()
# volume_matrix = df_price['5. volume'].as_matrix()

print('Creating Training Dataset')
training_set = df_price.iloc[:, 1:2].values
sc = MinMaxScaler(feature_range=(0,1))
scaled_training_set = sc.fit_transform(training_set)

x_train = []
y_train = []

for i in range(60,2035):
    x_train.append(scaled_training_set[i-60:i, 0])
    y_train.append(scaled_training_set[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))



print('Creating Model')
# Create Model
model = Sequential()

model.add(LSTM(units=50,return_sequences=True,input_shape=(x_train.shape[1], 1)))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50,return_sequences=True))
model.add(Dropout(0.2))

model.add(LSTM(units=50))
model.add(Dropout(0.2))

model.add(Dense(units=1))

print('Compiling Model')
model.compile(optimizer='adam',loss='mean_squared_error')

print('Training Model')
model.fit(x_train,y_train,epochs=3,batch_size=32)

# Create Test Set
testing_set = df_price.iloc[:, 1:2].values
dataset_total = pd.concat((training_set['1. open'], testing_set['1. open']))
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60, 76):
    X_test.append(inputs[i-60:i, 0])
X_test = np.array(X_test)
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predicted_stock_price = model.predict(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

plt.plot(real_stock_price, color = 'black', label = stock)
plt.plot(predicted_stock_price, color = 'green', label = ('Predicted ' + stock + 'Stock Price'))
plt.title('Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel(stock + ' Stock Price')
plt.legend()
plt.show()

# fig = go.Figure()
# # Adds Price Line to Figure
# fig.add_trace(go.Scatter(x=df_price['date'],
#                         y=df_price['4. close'],
#                         name=stock + " Adjusted Daily Close Price",
#                         line_color='red'))

# fig.add_trace(go.Scatter(x=df_price['date'],
#                         y=df_price['1. open'],
#                         name=stock + " Adjusted Daily Close Price",
#                         line_color='deepskyblue'))
                    
# fig.add_trace(go.Scatter(x=df_price['date'],
#                         y=df_price['2. high'],
#                         name=stock + " Adjusted Daily Close Price",
#                         line_color='green'))                   

# fig.add_trace(go.Scatter(x=df_price['date'],
#                         y=df_price['3. low'],
#                         name=stock + " Adjusted Daily Close Price",
#                         line_color='blue'))  

# # Adds Indicator Lines to Figure
# for i in range(len(df_ind)):                        
#     fig.add_trace(go.Scatter(x=df_ind[i]['date'],
#                             y=df_ind[i][indicators[i]],
#                             name=stock + indicators[i],
#                             line_color='orange'))
# # Update Figure Context              
# fig.update_traces(
#     hoverinfo="y+text",
#     line={"width": 0.5},
#     marker={"size": 8},
#     mode="lines",
#     showlegend=False
# )
# # Update Figure Layout
# fig.update_layout(
#     title_text='Time Series with Rangeslider',
#     xaxis_rangeslider_visible=True)

# fig.show()

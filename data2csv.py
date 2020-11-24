from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
from pprint import pprint
import json
import argparse
from av_key import av_key
from sklearn import preprocessing
import numpy as np
import pandas as pd


prior_days = 50

# stock = symbol
# time_window = time frame for dataset (daily (YTD), intradaily (Week), etc.)
# indicators[] = array of indicators
def save_csv(stock, time_window, indicators):
    api_key = av_key
    print(stock, time_window)
    ts = TimeSeries(key=api_key, output_format='pandas')
    ti = TechIndicators(key=api_key, output_format='pandas')

    if time_window.upper() == 'DAILY':
        data, meta_data = ts.get_daily(symbol=stock, outputsize='full')
    elif time_window.upper() == 'INTRADAILY':
        data, meta_data = ts.get_intraday(symbol=stock, interval='1min', outputsize='full')
    else:
        print('Invalid Time Window.')
        return

    data = data.sort_values(by=['date'], ascending=True)
    pprint(data.head(10))
    data.to_csv(f'./charts/{stock}_{time_window.lower()}.csv')

    for indicator in indicators:
        # Daily values (throughout the past year)
        if time_window.upper() == 'DAILY':
            if indicator.upper() == 'SMA':
                data, meta_data = ti.get_sma(symbol=stock, interval='daily', series_type='close')
            elif indicator.upper() == 'RSI':
                data, meta_data = ti.get_rsi(symbol=stock, interval='daily', series_type='close')
            elif indicator.upper() == 'MACD':
                data, meta_data = ti.get_macd(symbol=stock, interval='daily', series_type='close')
        # Intradaily values (throughout the past week) 
        else:
            if indicator.upper() == 'SMA':
                data, meta_data = ti.get_sma(symbol=stock, interval='1min', series_type='close')
            elif indicator.upper() == 'RSI':
                data, meta_data = ti.get_rsi(symbol=stock, interval='1min', series_type='close')
            elif indicator.upper() == 'MACD':
                data, meta_data = ti.get_macd(symbol=stock, interval='1min', series_type='close')  

        data = data.sort_values(by=['date'], ascending=True)
        pprint(data.head(10))
        data.to_csv(f'./charts/{stock}_{time_window.lower()}_{indicator.lower()}.csv')

def save_dataset(csv):
    data = pd.read_csv(csv)
    closing = data['4. close']
    # data = data.drop('date', axis=1)
    # data = data.drop(0, axis=0)
    scl = preprocessing.MinMaxScaler()
    closing = closing.values.reshape(closing.shape[0],1)
    closing = scl.fit_transform(closing)
    print(closing)
    #data_normalised = data.reshape(data_normalised.shape[0],1)
    #data_normalised = data_normaliser.fit_transform(data)

    def processData(data, lb):
        X, Y = [], []
        for i in range(len(data)-lb-1):
            X.append(data[i:(i+lb), 0])
            Y.append(data[(i+lb), 0])
        return np.array(X), np.array(Y)

    X, Y = processData(closing, 7)
    X_train,X_test = X[:int(X.shape[0]*0.80)],X[int(X.shape[0]*0.80):]
    Y_train,Y_test = Y[:int(Y.shape[0]*0.80)],Y[int(Y.shape[0]*0.80):]

    print(X_train.shape[0])
    print(X_test.shape[0])
    print(Y_train.shape[0])
    print(Y_test.shape[0])
    
    assert X_train.shape[0] == Y_train.shape[0] and X_test.shape[0] == Y_test.shape[0]
    return X_train, X_test, Y_train, Y_test, scl, data['4. close']


# def save_dataset(csv):
#     data = pd.read_csv(csv)
#     # drop date row, so we can scale day by day
#     data = data.drop('date', axis=1)
#     data = data.drop(0, axis=0)

#     data = data.values

#     # Scale between 0 to 1 -> data_normalised
#     data_normaliser = preprocessing.MinMaxScaler()
#     data_normalised = data_normaliser.fit_transform(data)

#     # Prepare Dataset for Model
#     # Using the amount of prior_days open, close, high, low, and volume. Predict the next open and close value.
#     ohlcv_histories_normalised = np.array([data_normalised[i:i + prior_days].copy() for i in range(len(data_normalised) - prior_days)])
    
#     # normalise next day predictions
#     next_day_open_values_normalised = np.array([data_normalised[:, 0][i + prior_days].copy() for i in range(len(data_normalised) - prior_days)])
#     next_day_open_values_normalised = np.expand_dims(next_day_open_values_normalised, -1)
#     next_day_open_values = np.array([data[:, 0][i + prior_days].copy() for i in range(len(data) - prior_days)])
#     next_day_open_values = np.expand_dims(next_day_open_values, -1)

#     next_day_close_values_normalised = np.array([data_normalised[:, 1][i + prior_days].copy() for i in range(len(data_normalised) - prior_days)])
#     next_day_close_values_normalised = np.expand_dims(next_day_close_values_normalised, -1)
#     next_day_close_values = np.array([data[:, 1][i + prior_days].copy() for i in range(len(data) - prior_days)])
#     next_day_close_values = np.expand_dims(next_day_close_values, -1)

#     y_normaliser = preprocessing.MinMaxScaler()
#     y_normaliser.fit(next_day_close_values)

#     # Basic technical indicators for now
#     def calc_ema(values, time_period):
#         # https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp
#         sma = np.mean(values[:, 3])
#         ema_values = [sma]
#         k = 2 / (1 + time_period)
#         for i in range(len(his) - time_period, len(his)):
#             close = his[i][3]
#             ema_values.append(close * k + ema_values[-1] * (1 - k))
#         return ema_values[-1]

#     technical_indicators = []
#     for his in ohlcv_histories_normalised:
#         # note since we are using his[3] we are taking the SMA of the closing price
#         sma = np.mean(his[:, 3])
#         macd = calc_ema(his, 12) - calc_ema(his, 26)
#         technical_indicators.append(np.array([sma]))
#         # technical_indicators.append(np.array([sma,macd,]))

#     technical_indicators = np.array(technical_indicators)

#     tech_ind_scaler = preprocessing.MinMaxScaler()
#     technical_indicators_normalised = tech_ind_scaler.fit_transform(technical_indicators)

#     assert ohlcv_histories_normalised.shape[0] == next_day_open_values_normalised.shape[0] == next_day_close_values_normalised.shape[0] == technical_indicators_normalised.shape[0]
#     return ohlcv_histories_normalised, technical_indicators_normalised, next_day_open_values_normalised, next_day_open_values, next_day_close_values_normalised, next_day_close_values, y_normaliser
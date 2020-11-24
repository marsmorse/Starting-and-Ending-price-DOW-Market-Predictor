import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from alpha_vantage.techindicators import TechIndicators
import matplotlib.pyplot as plt
from av_key import av_key

stock = input('Stock: ')
print('')

api_key = av_key
def rsi_dataframe(stock=stock):

    period = 60
    ts = TimeSeries(key=api_key, output_format='pandas')
    data_ts = ts.get_intraday(stock.upper(), interval='1min', outputsize='full')

    ti = TechIndicators(key=stock.upper(), output_format='pandas')
    
    #RSI determines if a stock is overbought or oversold
    data_ti, meta_data_ti = ti.get_rsi(symbol=stock.upper(), interval='1min', time_period=period, series_type='close')
    df_ti = data_ti
    #SMA
    data_sma, meta_data_sma = ti.get_sma(symbol=stock.upper(), interval='1min', time_period=period, series_type='close')
    df_sma = data_sma.iloc[1::] # since sma start with one before

    #ensure indexes are the same
    df_ti.index = df_sma.index    
    
    fig, ax1 = plt.subplots()
    ax1.plot(df_ti, 'b-')
    ax2 = ax1.twinx() #plots along same x value
    ax2.plot(df_sma, 'r-')
    plt.title("SMA & RSI Graph")
    plt.show()







    # df = data_ts[0][period::]

    # # df.index = pd.Index(map(lambda x: str(x)[:-3], df.index))

    # df2 = data_ti


    # total_df = pd.merge(df,  df2, on="date")
    # print(total_df)
    
    return 

rsi_dataframe()
# ts = TimeSeries(key=api_key, output_format='pandas')
# data_ts, meta_data_ts = ts.get_intraday(symbol='MSFT', interval='1min', outputsize='full')

# period = 60

# ti = TechIndicators(key=api_key, output_format='pandas')
# data_ti, meta_data_ti = ti.get_sma(symbol='MSFT', interval='1min',
#                                     time_period=period, series_type='close')

# df1 = data_ti
# df2 = data_ts['4. close'].iloc[period-1::]

# df2.index = df1.index

# total_df = pd.concat([df1, df2], axis=1)
# print(total_df)

# total_df.plot()
# plt.show()

import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Dense, Dropout, LSTM, Input, Activation
from keras import optimizers
import numpy as np
from data2csv import save_dataset, prior_days

np.random.seed(4)
tf.random.set_seed(4)

# Get dataset returned from save_dataset
ohlcv_histories_normalised, _, next_day_open_values_normalised, next_day_open_values, next_day_close_values_normalised, next_day_close_values, y_normaliser = save_dataset(f'./charts/AAPL_daily.csv')
  
# Test Split (Percentage of Data to use as Training vs Testing)
test_split = 0.9
n = int(ohlcv_histories_normalised.shape[0] * test_split)

# Split Training Data
ohlcv_train = ohlcv_histories_normalised[:n]
y_train_open = next_day_open_values_normalised[:n]
y_train_close = next_day_close_values_normalised[:n]

# Split Testing Data
ohlcv_test = ohlcv_histories_normalised[n:]
y_test_open = next_day_open_values_normalised[n:]
y_test_close = next_day_close_values_normalised[n:]

max_y_test = next_day_close_values[n:]

print(ohlcv_train.shape)
print(ohlcv_test.shape)

# Define Model Architecture
# //////////////////////////////////////////////////////////
# INPUT LAYER: BATCHES OF PRIOR DAYS, 5 INPUTS (OHLCV)
lstm_input = Input(shape=(prior_days, 5), name='lstm_input')
# FIRST HIDDEN LAYER
x = LSTM(50, name='lstm_0')(lstm_input)
# DROPOUT FOR FIRST HIDDEN LAYER
x = Dropout(0.2, name='lstm_dropout_0')(x)
x = Dense(64, name='dense_0')(x)
x = Activation('sigmoid', name='sigmoid_0')(x)
x = Dense(1, name='dense_1')(x)
# OUTPUT LAYER
output = Activation('linear', name='linear_output')(x)
# //////////////////////////////////////////////////////////

# Create Model
model = Model(inputs=lstm_input, outputs=output)
adam = optimizers.Adam(lr=0.0005)
model.compile(optimizer=adam, loss='mse')
model.fit(x=ohlcv_train, y=y_train_close, batch_size=32, epochs=10, shuffle=False, validation_split=0.1)

# Predict
y_test_predicted = model.predict(ohlcv_test)
y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
y_predicted = model.predict(ohlcv_histories_normalised)
y_predicted = y_normaliser.inverse_transform(y_predicted)

assert max_y_test.shape == y_test_predicted.shape
real_mse = np.mean(np.square(max_y_test - y_test_predicted))
scaled_mse = real_mse / (np.max(max_y_test) - np.min(max_y_test)) * 100
print(scaled_mse)


# Plot
import matplotlib.pyplot as plt

plt.gcf().set_size_inches(22, 15, forward=True)

start = 0
end = -1

real = plt.plot(max_y_test[start:end], label='real')
pred = plt.plot(y_test_predicted[start:end], label='predicted')
plt.legend(['Real', 'Predicted'])

plt.show()

from datetime import datetime
model.save(f'basic_lstm_model.h5')

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler


data = pd.read_csv('IoT.csv')

print(data.head)

data = pd.get_dummies(data, columns=['Company'])
scaler = MinMaxScaler()

scaler = MinMaxScaler()


print(data.columns)  
print(data[['Market Share', 'Units Sold', 'Revenue', 'YoY Growth', 'Sales', 'CVP']].dtypes)

scaled_data = scaler.fit_transform(data[['Market Share', 'Units Sold', 'Revenue', 'YoY Growth', 'Sales', 'CVP']])


data[['Market Share', 'Units Sold', 'Revenue', 'YoY Growth', 'Sales', 'CVP']] = scaled_data

print(data.head())

data.set_index('Year', inplace=True) 

print(data.head()) 



def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step):
        X.append(data[i:(i + time_step), :])
        y.append(data[i + time_step, 0])      
    return np.array(X), np.array(y)


features = data[['Market Share', 'Units Sold', 'Revenue', 'YoY Growth', 'Sales', 'CVP']].values
time_step = 3
X, y = create_dataset(features, time_step)


X = X.reshape(X.shape[0], X.shape[1], X.shape[2])

print(X.shape, y.shape)  

train_size = int(len(X) * 0.8)  
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print(X_train.shape, X_test.shape)  



model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, X.shape[2])))  
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=False))  
model.add(Dense(25))  
model.add(Dense(1))   


model.compile(optimizer='adam', loss='mean_squared_error')


history = model.fit(X_train, y_train, batch_size=32, epochs=50, validation_data=(X_test, y_test))





predictions = model.predict(X_test)


predictions_reshaped = np.zeros((predictions.shape[0], X.shape[2]))


predictions_reshaped[:, 0] = predictions.flatten()


predictions = scaler.inverse_transform(predictions_reshaped)


y_test_reshaped = np.zeros((y_test.shape[0], X.shape[2]))
y_test_reshaped[:, 0] = y_test.flatten()


y_test_rescaled = scaler.inverse_transform(y_test_reshaped)


plt.figure(figsize=(10,5))
plt.plot(y_test_rescaled[:, 0], label='True Market Share', color='blue')
plt.plot(predictions[:, 0], label='Predicted Market Share', color='red')
plt.title('Market Share Prediction using LSTM')
plt.xlabel('Years')
plt.ylabel('Market Share (%)')
plt.legend()
plt.grid(True)
plt.show()




rmse = np.sqrt(mean_squared_error(y_test, predictions[:, 4]))
print(f'Root Mean Squared Error: {rmse}')
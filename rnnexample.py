import matplotlib.pyplot as plt
import matplotlib
import tensorflow as tf
import numpy as np
import pandas as pd
import datetime
import os
import datetime
from sklearn.preprocessing import MinMaxScaler
from tensorflow.python import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Input, Dense, GRU, Embedding, LSTM, Dropout
from tensorflow.python.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, ReduceLROnPlateau

class CustomHistory(keras.callbacks.Callback):
    def init(self):
        self.train_loss = []
        self.val_loss = []

    def on_epoch_end(self, batch, logs={}):
        self.train_loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
#약 30분 후의 주차장 내 차량 숫자 예측하기
#랜덤에 의해 똑같은 결과를 재현하도록 시드 설정
tf.set_random_seed(777)
print(tf.__version__)
print(tf.keras.__version__)
print(pd.__version__)

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)
def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()
def reverse_min_max_scaler(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1e-7)) + org_x_np.min()

#데이터 로딩
file_name = 'park_daily.csv'
encoding = 'euc_kr'
name = ['max', 'parkinglot', 'date']
raw_dataframe = pd.read_csv(file_name, names=name, encoding=encoding)
raw_dataframe.info()

#필요없는 요소들 삭제
del raw_dataframe['date']
del raw_dataframe['max']

car_info = raw_dataframe.values[0:].astype(np.int)
print("car_info.shape: ", car_info.shape)
print("car_info[0]", car_info[0])

#정규화
norm_car = MinMaxScaler(car_info)
print("car_info[0]: ", car_info[0])
print("norm_car[0]:", norm_car[0])

#정규화 그래프 그려보기
plot_x = np.arange(len(car_info))
plot_y = norm_car
plt.plot(plot_x, plot_y)
plt.show()

def create_dataset(car_data, look_back=1):
    dataX, dataY = [], []
    for i in range(len(car_data)-look_back):
        dataX.append(car_data[i:(i+look_back), 0])
        dataY.append(car_data[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 36

#데이터 분리
num = len(norm_car)
train = norm_car[0:int(num*0.5)]
val = norm_car[int(num*0.5): int(num*0.7)]
test = norm_car[int(num*0.7):]

x_train, y_train = create_dataset(train, look_back)
x_val, y_val = create_dataset(val, look_back)
x_test, y_test = create_dataset(test, look_back)

# 데이터셋 전처리
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))


###########################################################
# # 2. 다층 퍼셉트론 모델 구성하기
# x_train = np.squeeze(x_train)
# x_val = np.squeeze(x_val)
# x_test = np.squeeze(x_test)
#
# model = Sequential()
# model.add(Dense(32, input_dim=36, activation="relu"))
# model.add(Dropout(0.3))
# for i in range(2):
#     model.add(Dense(32, activation="relu"))
#     model.add(Dropout(0.3))
# model.add(Dense(1))
#
# # 3. 모델 학습과정 설정하기
# model.compile(loss='mean_squared_error', optimizer='adagrad')
#
# # 4. 모델 학습시키기
# hist = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_val, y_val))
#
# # 5. 학습과정 살펴보기
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.ylim(0.0, 0.15)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# # 6. 모델 평가하기
# trainScore = model.evaluate(x_train, y_train, verbose=0)
# print('Train Score: ', trainScore)
# valScore = model.evaluate(x_val, y_val, verbose=0)
# print('Validataion Score: ', valScore)
# testScore = model.evaluate(x_test, y_test, verbose=0)
# print('Test Score: ', testScore)
#
# # 7. 모델 사용하기
# look_ahead = 250
# xhat = x_test[0, None]
# predictions = np.zeros((look_ahead, 1))
# for i in range(look_ahead):
#     prediction = model.predict(xhat, batch_size=32)
#     predictions[i] = prediction
#     xhat = np.hstack([xhat[:, 1:], prediction])
#
# plt.figure(figsize=(12, 5))
# plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
# plt.plot(np.arange(look_ahead), y_test[:look_ahead], label="test function")
# plt.legend()
# plt.show()

##################################################################################

# #순환신경망 모델
#
# # 2. 모델 구성하기
# model = Sequential()
# model.add(LSTM(32, input_shape=(None, 1)))
# model.add(Dropout(0.3))
# model.add(Dense(1))
#
# # 3. 모델 학습과정 설정하기
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# # 4. 모델 학습시키기
# hist = model.fit(x_train, y_train, epochs=200, batch_size=32, validation_data=(x_val, y_val))
#
# # 5. 학습과정 살펴보기
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.ylim(0.0, 0.15)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# # 6. 모델 평가하기
# trainScore = model.evaluate(x_train, y_train, verbose=0)
# model.reset_states()
# print('Train Score: ', trainScore)
# valScore = model.evaluate(x_val, y_val, verbose=0)
# model.reset_states()
# print('Validataion Score: ', valScore)
# testScore = model.evaluate(x_test, y_test, verbose=0)
# model.reset_states()
# print('Test Score: ', testScore)
#
# # 7. 모델 사용하기
# look_ahead = 250
# xhat = x_test[0]
# predictions = np.zeros((look_ahead, 1))
# for i in range(look_ahead):
#     prediction = model.predict(np.array([xhat]), batch_size=1)
#     predictions[i] = prediction
#     xhat = np.vstack([xhat[1:], prediction])
#
# plt.figure(figsize=(12, 5))
# plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
# plt.plot(np.arange(look_ahead), y_test[:look_ahead], label="test function")
# plt.legend()
# plt.show()

################################################################################

#상태유지 신경망 모델

# 2. 모델 구성하기
model = Sequential()
model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
model.add(Dropout(0.3))
model.add(Dense(1))

# 3. 모델 학습과정 설정하기
model.compile(loss='mean_squared_error', optimizer='adam')

# 4. 모델 학습시키기
custom_hist = CustomHistory()
custom_hist.init()

for i in range(200):
    model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist],
              validation_data=(x_val, y_val))
    model.reset_states()

# 5. 학습과정 살펴보기
plt.plot(custom_hist.train_loss)
plt.plot(custom_hist.val_loss)
plt.ylim(0.0, 0.15)
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# 6. 모델 평가하기
trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
model.reset_states()
print('Train Score: ', trainScore)
valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
model.reset_states()
print('Validataion Score: ', valScore)
testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
model.reset_states()
print('Test Score: ', testScore)

# 7. 모델 사용하기
look_ahead = 250
xhat = x_test[0]
predictions = np.zeros((look_ahead, 1))
for i in range(look_ahead):
    prediction = model.predict(np.array([xhat]), batch_size=1)
    predictions[i] = prediction
    xhat = np.vstack([xhat[1:], prediction])

plt.figure(figsize=(12, 5))
plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
plt.plot(np.arange(look_ahead), y_test[:look_ahead], label="test function")
plt.legend()
plt.show()

################################################################################

# #상태유지 스택 순환신경망 모델
# # 2. 모델 구성하기
# model = Sequential()
# for i in range(2):
#     model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True, return_sequences=True))
#     model.add(Dropout(0.3))
# model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
# model.add(Dropout(0.3))
# model.add(Dense(1))
#
# # 3. 모델 학습과정 설정하기
# model.compile(loss='mean_squared_error', optimizer='adam')
#
# # 4. 모델 학습시키기
# custom_hist = CustomHistory()
# custom_hist.init()
#
# for i in range(200):
#     model.fit(x_train, y_train, epochs=1, batch_size=1, shuffle=False, callbacks=[custom_hist],
#               validation_data=(x_val, y_val))
#     model.reset_states()
#
# # 5. 학습과정 살펴보기
# plt.plot(custom_hist.train_loss)
# plt.plot(custom_hist.val_loss)
# plt.ylim(0.0, 0.15)
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'val'], loc='upper left')
# plt.show()
#
# # 6. 모델 평가하기
# trainScore = model.evaluate(x_train, y_train, batch_size=1, verbose=0)
# model.reset_states()
# print('Train Score: ', trainScore)
# valScore = model.evaluate(x_val, y_val, batch_size=1, verbose=0)
# model.reset_states()
# print('Validataion Score: ', valScore)
# testScore = model.evaluate(x_test, y_test, batch_size=1, verbose=0)
# model.reset_states()
# print('Test Score: ', testScore)
#
# # 7. 모델 사용하기
# look_ahead = 250
# xhat = x_test[0]
# predictions = np.zeros((look_ahead, 1))
# for i in range(look_ahead):
#     prediction = model.predict(np.array([xhat]), batch_size=1)
#     predictions[i] = prediction
#     xhat = np.vstack([xhat[1:], prediction])
#
# plt.figure(figsize=(12, 5))
# plt.plot(np.arange(look_ahead), predictions, 'r', label="prediction")
# plt.plot(np.arange(look_ahead), y_test[:look_ahead], label="test function")
# plt.legend()
# plt.show()

#############################################################################

# #다층 퍼셉트론 모델
# model = Sequential()
# model.add(Dense(32,input_dim=40,activation="relu"))
# model.add(Dropout(0.3))
# for i in range(2):
#     model.add(Dense(32,activation="relu"))
#     model.add(Dropout(0.3))
# model.add(Dense(1))

#
# #순환신경망 모델
# model = Sequential()
# model.add(LSTM(32, input_shape=(None, 1)))
# model.add(Dropout(0.3))
# model.add(Dense(1))

#
# #상태유지 순환신경망모델
# model = Sequential()
# model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
# model.add(Dropout(0.3))
# model.add(Dense(1))
#

# #상태유지 스택 순환신경망 모델
# model = Sequential()
# for i in range(2):
#     model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True, return_sequences=True))
#     model.add(Dropout(0.3))
# model.add(LSTM(32, batch_input_shape=(1, look_back, 1), stateful=True))
# model.add(Dropout(0.3))
# model.add(Dense(1))





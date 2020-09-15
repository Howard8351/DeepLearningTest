#用numpy實作RNN基本概念流程
import numpy as np
#取樣間格
timesteps = 100
#輸入資料維度
input_features = 32
#輸出資料維度
output_features = 64
#建立隨機輸入資料
inputs = np.random.random((timesteps, input_features))
#初始化前一次的輸出狀態
state_t = np.zeros((output_features,))
#隨機建立權重矩陣
W = np.random.random((output_features, input_features))
U = np.random.random((output_features, output_features))
b = np.random.random((output_features,))
successive_outputs = []
#將輸入帶入權重矩陣
for input_t in inputs:
    output_t = np.tanh(np.dot(W, input_t) + np.dot(U, state_t) + b)
    successive_outputs.append(output_t)
    state_t = output_t
final_output_sequence = np.concatenate(successive_outputs, axis=0)

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
max_features = 10000
maxlen = 500
batch_size = 32
print('Loading data...')
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words = max_features)
print(len(input_train), 'train sequences')
print(len(input_test), 'test sequences')
print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train, maxlen=maxlen)
input_test = sequence.pad_sequences(input_test, maxlen=maxlen)
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

#用keras實作基本RNN
from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
model = Sequential([
    Embedding(10000, 32),
    SimpleRNN(32)
    ])
model.summary()

#SimpleRNN的return_sequences 會決定是否要回每一個輸入相對應的輸出
model = Sequential([
    Embedding(10000, 32),
    SimpleRNN(32, return_sequences=True)
    ])
model.summary()

#也可以把多個RNN堆疊在一起以強化網路的代表性
#但是前面幾層的RNN都必須回傳全部的輸出
model = Sequential([
    Embedding(10000, 32),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(32, return_sequences=True),
    SimpleRNN(32)
    ])
model.summary()

#建立一個RNN來做IMDB評論分類
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
max_features = 10000
maxlen = 500
batch_size = 32
#匯入資料
(input_train, y_train), (input_test, y_test) = imdb.load_data(num_words = max_features)
#把輸入資料轉成固定長度的tensor
input_train = sequence.pad_sequences(input_train, maxlen=maxlen, truncating = 'post')
input_test = sequence.pad_sequences(input_test, maxlen=maxlen, truncating = 'post')
print('input_train shape:', input_train.shape)
print('input_test shape:', input_test.shape)

from tensorflow.keras.layers import Dense, Embedding, SimpleRNN
from tensorflow.keras.models import Sequential
#將資料透過Embedding學習後直接匯入RNN
model = Sequential([
    Embedding(max_features, 32),
    SimpleRNN(32),
    Dense(1, activation = 'sigmoid')
])

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

#LSTM是SampleRNN的改良版
#可以避免較早期訓練的結果逐漸失去影響力
from tensorflow.keras.layers import LSTM
model = Sequential([
    Embedding(max_features, 32),
    LSTM(32),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(input_train, y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')

plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
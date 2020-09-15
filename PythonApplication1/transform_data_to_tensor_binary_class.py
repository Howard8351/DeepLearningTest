import tensorflow as tf
import tensorflow.keras.datasets.imdb as imdb
import numpy as np
import matplotlib.pyplot as plt

#使用One-hot encode 把數值序列轉成一個稀疏矩陣，以利運算
#初始維度預設為10000
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results
    



#從imdb匯入電影評論資料
#只拿出現頻率較高的10000個單字
(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
print(len(train_data))
print(len(test_data))


#取得單字和其整數編碼的字典(dict)
word_index = imdb.get_word_index()
#將字典中的單字和數值反轉後建立新的字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
print('train_data:', train_data[0])
#使用新的字典將第一筆資料從數值編碼還原
decoded_review = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print('decoded_review:', decoded_review)


print(type(train_data))
print(type(train_data[0]))
print((train_data[0][0]))
print(train_data.shape)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)
print(type(x_train))
print(x_train.shape)
print(x_train[1,])

print(type(train_labels[0]))
y_train = np.asarray(train_labels).astype('float32')
print(type(y_train[0]))
y_test = np.asarray(test_labels).astype('float32')

#建立預測模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1,  activation='sigmoid')
])
#設定模型的最佳化和結果評估方式
model.compile(optimizer= tf.keras.optimizers.RMSprop(lr = 0.001),
              loss='binary_crossentropy',metrics=['accuracy'])
#切出訓練和驗證資料集
x_val = x_train[:10000]
partial_x_train = x_train[10000:]
y_val = y_train[:10000]
partial_y_train = y_train[10000:]

#訓練並保存結果
history = model.fit(partial_x_train, partial_y_train,
                    epochs=20, batch_size=512, validation_data=(x_val, y_val))

#取出結果並繪圖
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
epochs = range(1, len(history_dict['accuracy']) + 1)
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


#調整參數後再從新建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(10000,)),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1,  activation='sigmoid')
])

model.compile(optimizer= tf.keras.optimizers.RMSprop(),
              loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train, y_train, epochs=4, batch_size=512)
results = model.evaluate(x_test, y_test)
results

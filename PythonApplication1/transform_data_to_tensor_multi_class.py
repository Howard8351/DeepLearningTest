import tensorflow as tf
import tensorflow.keras.datasets.reuters as reuters
import numpy as np
import matplotlib.pyplot as plt

#使用One-hot encode 把數值序列轉成一個稀疏矩陣，以利運算
#初始維度預設為10000
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

#使用One-hot encode 把資料類別序列轉成一個稀疏矩陣，以利運算
#初始維度預設為46(代表有46種類別)
def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


#從reuters匯入文章類別資料集
#只拿出現頻率較高的10000個單字
(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)
print(len(train_data))
print(len(test_data))
#input('按Enter以繼續')

#取得單字和其整數編碼的字典(dict)
word_index = reuters.get_word_index()
#將字典中的單字和數值反轉後建立新的字典
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])
#使用新的字典將第一筆資料從數值編碼還原
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
print('decoded_newswire:', decoded_newswire)
#input('按Enter以繼續')

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

#建立文章類別分類模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10000,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(46,  activation='softmax')
])

#設定模型的最佳化和結果評估方式
model.compile(optimizer='rmsprop', loss='categorical_crossentropy'
              , metrics=['accuracy'])

#切出訓練和驗證資料集
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

#訓練並保存結果
history = model.fit(partial_x_train, partial_y_train, epochs=20
                    , batch_size=512, validation_data=(x_val, y_val))

#取出結果並繪圖
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.clf()
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
input('按Enter以繼續')

#調整參數後再從新建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10000,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(46,  activation='softmax')
])

model.compile(optimizer='rmsprop', loss='categorical_crossentropy'
              , metrics=['accuracy'])

model.fit(partial_x_train, partial_y_train, epochs=9
                    , batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, one_hot_test_labels)
print(results)
predictions = model.predict(x_test)
predictions[0].shape
np.sum(predictions[0])
np.argmax(predictions[0])
input('按Enter以繼續')

y_train = np.array(train_labels)
y_test = np.array(test_labels)
x_val = x_train[:1000]
partial_x_train = x_train[1000:]
y_val = y_train[:1000]
partial_y_train = y_train[1000:]

#調整參數後再從新建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(10000,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(46,  activation='softmax')
])

model.compile(optimizer='rmsprop', loss='sparse_categorical_crossentropy'
              , metrics=['accuracy'])

model.fit(partial_x_train, partial_y_train, epochs=9
                    , batch_size=512, validation_data=(x_val, y_val))
results = model.evaluate(x_test, y_test)
results

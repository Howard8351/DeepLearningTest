
#tensorflow 基本分類
print('tensorflow 基本分類')
import tensorflow as tf
import matplotlib.pyplot as plt
#匯入資料
print('匯入資料(此為服裝資料集)')
fashion_mnist = tf.keras.datasets.fashion_mnist
#對資料做正規化
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images / 255.0
test_images = test_images / 255.0

#建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#定義最佳化和驗證模型的方式
model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#對模型作訓練
model.fit(train_images, train_labels, epochs=5)

model2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128),
    tf.keras.layers.PReLU(),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model2.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model2.fit(train_images, train_labels, epochs=5)

import math
import numpy as np
tf.keras.backend.set_floatx('float64')

#設定最佳化方式
optimizer = tf.keras.optimizers.Adam()
#設定損失函數
loss = tf.keras.losses.SparseCategoricalCrossentropy()
batch_size = 32
for epoch in range(5):
    start_index = 0
    end_index = 0
    for i in range(math.floor(len(train_images) / batch_size) + 1):
        if (start_index + batch_size) > len(train_images):
            end_index = len(train_images)
        else:
            end_index = start_index + batch_size
        batch_data = train_images[start_index:end_index,]
        batch_label = train_labels[start_index:end_index,]
        #batch_label = np.reshape(batch_label, (1,1, len(batch_label)))
        with tf.GradientTape() as tape:
            #train_model_output = tf.math.argmax(model2(batch_data), axis = 1)
            train_model_output = model2(batch_data)
            loss_value = loss(batch_label, train_model_output)
        #計算梯度
        gradients = tape.gradient(loss_value, model2.trainable_weights)
        #根據梯度更新權重
        optimizer.apply_gradients(zip(gradients, model2.trainable_weights))
        start_index += batch_size
    a, b = model2.evaluate(train_images, train_labels)
    #print('Loss: {}' .format(loss(train_labels, model_predict).numpy()))
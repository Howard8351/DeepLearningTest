import tensorflow as tf

#從mist匯入資料
(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

print(train_images.shape)
input('輸入Enter以繼續')

#資料為灰階影像
#將資料正規畫成0~1之間
train_images = train_images / 255.0
test_images = test_images / 255.0

#建立模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#設定最佳化參數
model.compile(optimizer='rmsprop', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#對模型做訓練
model.fit(train_images, train_labels, epochs=5)
#用訓練資料評估模型
model.evaluate(test_images,test_labels)



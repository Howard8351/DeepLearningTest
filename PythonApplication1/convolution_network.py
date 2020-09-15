import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model

#Conv2D只接受3維輸入資料(高度,寬度,層數)
#層數可視為影像由幾種資料組成
# 如:灰階影性只有一張亮度值，層數為1
#而RGB影像由三原色組成,層數為3
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation= 'relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation= 'relu')
  ])

#window_size = 4
#input_layer = Input(shape = (window_size,window_size,3))
#con2d_layer1 = layers.Conv2D(10, (1,1), activation = 'relu')(input_layer)
#maxpool_layer1 = layers.MaxPooling2D((2,2))(con2d_layer1)
#flatten_layer = layers.Flatten()(maxpool_layer1)
#hidden_layer = layers.Dense(150, activation = 'relu')(flatten_layer)
#output_layer = layers.Dense(1, activation = 'sigmoid')(hidden_layer)
#model = Model(input_layer, output_layer)

print(model.summary())
input('輸入Enter以繼續')

#這兩種layer第二個參數可以只用一個整數代表所有維度的值
model = tf.keras.Sequential([
    layers.Conv2D(32, 3, activation= 'relu', input_shape=(None, None,1)),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation= 'relu'),
    layers.MaxPooling2D(2),
    layers.Conv2D(64, 3, activation= 'relu')
  ])

model.summary()

#當模型已經建立好後，可以用add來依序新增網路層
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

#匯入資料
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#將資料格式重建以符合卷積網路的輸入
#原始資料的shape是(60000, 28, 28)
#新增資料層數
train_images = train_images.reshape((60000, 28, 28, 1))
#將資料作正規化
train_images = train_images.astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
test_acc

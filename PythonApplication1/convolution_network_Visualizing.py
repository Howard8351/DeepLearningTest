#由於卷積網路是少數有明確學習規則的類神經網路
#所以可以藉由視覺化來了解卷積網路學習的特徵
from tensorflow.keras.models import load_model

#匯入之前建立的模型
model = load_model('cats_and_dogs_small_2.h5')
model.summary()

from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

#匯入圖片
img_path = 'E:/python program/dogs-vs-cats/dogs-vs-cats-resample/test/cats/cat.1700.jpg'
img = image.load_img(img_path, target_size=(150, 150))
#image.load_img 匯入的格式是img所以要轉成numpy array
img_tensor = image.img_to_array(img)
print(img_tensor.shape)
#轉換後是一個3維的tensor(長,寬,位元)
#但是這個卷積網路需要4維tensor輸入，所以我們在最前面加入第四維(第幾張圖)
img_tensor = np.expand_dims(img_tensor, axis=0)
print(img_tensor.shape)
img_tensor /= 255.


plt.imshow(img_tensor[0])
#plt.show()
plt.close()

from tensorflow.keras import models
#取出模型的前8層
#layer_outputs 為一個長度為8的list
layer_outputs = [layer.output for layer in model.layers[:8]]
#給定輸入和輸出格式建立新的模型
#這個寫法會把list中的8個內容的成8個輸出節點
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
#將影像匯入模型取的輸出
activations = activation_model.predict(img_tensor)
first_layer_activation = activations[0]
#確認第1個輸出的大小
print(first_layer_activation.shape)
#可以看到第一層的輸出是一張148*148*32的3維tensor

import matplotlib.pyplot as plt

#用迴圈試著畫出前5個特徵
for plotdata in first_layer_activation[0,:,:, range(5)]:
    plt.matshow(plotdata, cmap='viridis')
    plt.show()
    plt.close()

#取得各層的名稱
layer_names = []
for layer in model.layers[:8]:
    layer_names.append(layer.name)

#把每一層學到的特徵都畫出來
images_per_row = 16
for layer_name, layer_activation in zip(layer_names, activations):
    n_features = layer_activation.shape[-1]
    size = layer_activation.shape[1]
    n_cols = n_features // images_per_row
    display_grid = np.zeros((size * n_cols, images_per_row * size))
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = layer_activation[0,:, :,
                                             col * images_per_row +
                                             row]
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image, 0, 255).astype('uint8')
            display_grid[col * size : (col + 1) * size,
                         row * size : (row + 1) * size] = channel_image
    
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1],
                        scale * display_grid.shape[0]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='viridis')
    plt.show()
    plt.close()



from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf

#匯入模型
model = VGG16(weights='imagenet')
#匯入圖片並最前處理以便做分類
img_path = 'D:/python program/elephants.png'
img = image.load_img(img_path, target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

preds = model.predict(x)
#將預測結果輸出成人類能看得懂的格式
#只輸出機率最高的前3項
print('Predicted:', decode_predictions(preds, top=3)[0])
#這可以看出機率最高的類別是第幾項
print(np.argmax(preds[0]))
print(np.argmax(preds))

#已知分類結果維非洲象
#接下來要來看看分類器認為圖片中那些部分比較像目標(Grad CAM)
#和非洲象相關的的輸出是在第386行
african_elephant_index = 386
#取出卷積網路的最後一層
last_conv_layer = model.get_layer('block5_conv3')

#計算非洲象的輸出權重和最後一層卷積網路輸出權重的梯度
#由於從tensorflow2 開始增加了GradientTape API需要做修改才能計算兩個tensor之間的梯度
#tf.keras.models.Model 會根據給予的輸入和輸出建立一個網路(輸入和輸出可以用list來做成多輸入和輸出)
#搜尋funcational api會有較多說明
grad_model = tf.keras.models.Model([model.inputs], [last_conv_layer.output, model.output])
model.summary()
grad_model.summary()

with tf.GradientTape() as tape:

    conv_outputs, predictions = grad_model(x)
    #取得是非洲象的機率
    african_elephant_output = predictions[:, african_elephant_index]

#做tensor降維度
#conv_outputs 是一個(1,14,14,512)4維的tensor
#output 是一個(14,14,512)3維的tensor
output = conv_outputs[0]
print(np.min(output))
print(np.max(output))
#tensorflow2.0 開始只能對在GradientTape中被記錄過的tensor計算梯度
#計算帶入資料後是目標的機率和最後一層卷積網路特徵圖之間的梯度
grads = tape.gradient(african_elephant_output, conv_outputs)[0]
print(np.min(grads))
print(np.max(grads))
#將output和grads中有正數的部分轉成1其餘皆為0
gate_f = tf.cast(output > 0, 'float32')
gate_r = tf.cast(grads > 0, 'float32')
#將剛才計算的梯度乘上濾波器
#相當於只保留帶入資料後，最後一層卷積網特徵圖和剛才計算的梯度都有正數的部分
guided_grads = gate_f * gate_r * grads

#針對guided_grads中前兩個維度的資料(特徵map)取平均
weights = tf.reduce_mean(guided_grads, axis=(0, 1))

#建立一個大小跟特徵map相同的一矩陣
cam = np.ones(output.shape[0:2], dtype = np.float32)
#將每個特徵map乘上對應的權重後加總在剛才建立的矩陣
for i, w in enumerate(weights):

    cam += w * output[:, :, i]

import cv2
#cam = cv2.resize(cam.numpy(), (224, 224))
#將最後得到的特徵熱區畫出來
cam = np.maximum(cam, 0)
heatmap = (cam - cam.min()) / (cam.max() - cam.min())
plt.matshow(heatmap)
plt.show()
#將特徵熱區和原始影像結合
img = cv2.imread(img_path)
plt.matshow(img)
#plt.show()
plt.close()
heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
superimposed_img = (superimposed_img - superimposed_img.min()) / (superimposed_img.max() - superimposed_img.min())
plt.matshow(superimposed_img)
plt.show()

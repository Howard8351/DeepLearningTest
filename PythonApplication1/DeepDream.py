from tensorflow.keras.applications import inception_v3
from tensorflow.data import Dataset
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt


#匯入inception_v3網路模型
model = inception_v3.InceptionV3(weights='imagenet',include_top=False)

#model.summary()
#在deepdream中只對混和卷機網路輸出的層有興趣
#在inception_v3這些層的名稱是mixed0~mixed10
#選擇參與deepdream的層
layer_name = ['mixed10']
#model.get_layer(name).output 會從指定的 layer name取出該層和其上方連接的layer
layer_output = [model.get_layer(name).output for name in layer_name]
#建立新的模型來計算選到的層的輸出
dream_model = tf.keras.Model(inputs=model.input, outputs=layer_output)

dream_model.summary()

#根據給予的影像和模型計算相對的loss
def calc_loss(img, model):
  # Pass forward the image through the model to retrieve the activations.
  #將影像轉換成batch size 為1的資料
  img_batch = tf.expand_dims(img, axis=0)
  #將影像帶入模型以取得指定layer的輸出
  layer_activations = model(img_batch)
  if len(layer_activations) == 1:
    layer_activations = [layer_activations]

  losses = []
  for act in layer_activations:
    loss = tf.math.reduce_mean(act)
    losses.append(loss)

  return  tf.math.reduce_sum(losses)

#取的檔案名稱
def get_file_name(file_path):
    #以python內建的檔案路徑分隔符號為標準分割字串
    split_string = tf.strings.split(file_path, os.path.sep)
    #回傳檔案名稱
    return split_string[-1]

#將image傳成tensor
def decode_image(image):
    #將image轉成nuit8 channel為3的tensor
    image = tf.io.decode_image(image, channels = 3)
    ##將image轉成指定的data type，並且作正規化
    #image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.cast(image, tf.float32)
    return image

#路徑處理function
def process_path(file_path):
    file_name = get_file_name(file_path)
    #匯入檔案
    image = tf.io.read_file(file_path)
    image = decode_image(image)
    return file_name, image

#將影像轉為0~255的區間並改成uint8 data type
def deprocess(img):
  img = 255*(img + 1.0)/2.0
  return tf.cast(img, tf.uint8)


base_image_path = 'E:/python program/Deep dream test/*.jpg'
#base_image_path = 'E:/python program/dogs-vs-cats/test/*.jpg'
#根據給予的路徑建立一個包含所有符合條件檔案路徑的dataset
image_list = Dataset.list_files(base_image_path)
#Dataset.list_files  的輸出為帶有數個檔案路徑的eager_tensor

#for file in image_list.batch(2):
#    print(file)

#根據給予的function 建立 Mapdataset物件
#會再被調用時將dataset中的元素帶入function
#num_parallel_calls 用來決定在載入多個元素時是否要用平行運算
image_dataset = image_list.map(process_path, num_parallel_calls = tf.data.experimental.AUTOTUNE)
#for file_name, image in image_dataset.take(1):
#    plt.imshow(image)
#    plt.show()
    
#建立一個繼承自 tf.Module的 DeepDream class
class DeepDream(tf.Module):
  #class 初始化
  def __init__(self, model):
    self.model = model

  #用tf.function加速運算
  #tf.TensorSpec用來監控數入參數是否是指定的形狀和data type
  @tf.function(
      input_signature=(
        tf.TensorSpec(shape=[None,None,3], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
        tf.TensorSpec(shape=[], dtype=tf.float32),)
  )
  #被呼叫時的運算
  def __call__(self, img, steps, step_size):
      print("Tracing")
      loss = tf.constant(0.0)
      for n in tf.range(steps):
        with tf.GradientTape() as tape:
          #tf.GradientTape 預設只會監控 tf 相關變數
          #所以需要手動指定監控img
          tape.watch(img)
          loss = calc_loss(img, self.model)

        #計算loss 和輸入影像中每個像素點的梯度
        gradients = tape.gradient(loss, img)

        #梯度正規劃
        gradients /= tf.math.reduce_std(gradients) + 1e-8 
        
        #在梯度上升中要把loss 最大化讓輸入影像盡可能去觸發 layer
        #可以藉由把梯度加到輸入影像中來做到這點
        img = img + gradients*step_size
        img = tf.clip_by_value(img, -1, 1)

      return loss, img

#建立DeepDream model
deepdream = DeepDream(dream_model)

#定義執行迴圈
def run_deep_dream_simple(img, steps=100, step_size=0.01):
  #將影像帶入inception_v3 中的影像前處理
  #預期輸入是np array 或是 tf.tensor 且範圍是0~255(似乎只能匯入float 的 data type)
  #輸出為-1~1之間的實數tensor
  img = tf.keras.applications.inception_v3.preprocess_input(img)

  #將影像帶入deepdream
  loss, img = deepdream(img, steps, tf.constant(step_size))

  #將最後的影像轉為0~255的區間
  result = deprocess(img)

  return result

save_path = 'E:/python program/Deep dream test/'

for file_name, image in image_dataset.take(1):
    dream_img = run_deep_dream_simple(img = image,
                                      steps = 10, step_size = 0.1)
    plt.imshow(np.array(dream_img))
    plt.savefig(save_path + 'Output_layer_{}'.format(layer_name[0]) + '.png')
    plt.close()

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
#from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.data import Dataset
from scipy import ndimage
import matplotlib.pyplot as plt
import numpy as np
import os
import concurrent.futures
import time
import math

def image_smooth(image_list, original_data_path, smooth_data_path):
    for image_name in image_list:
        image_data = image.load_img(os.path.join(original_data_path, image_name))
        image_data = image.img_to_array(image_data)
        image_data = tf.image.resize(image_data, [300, 300]).numpy()  
        smooth_image= []
        for channel in range(np.shape(image_data)[-1]):
            smooth_image.append(ndimage.gaussian_filter(image_data[:,:, channel], 2))
        smooth_image = np.moveaxis(np.asarray(smooth_image), 0, -1)
        output_image = np.concatenate((image_data, smooth_image), axis = 1)

        image.save_img(os.path.join(smooth_data_path, image_name), output_image, data_format = 'channels_last')

def image_resize(image_list, original_data_path, resize_data_path):
    for image_name in image_list:
        image_data = image.load_img(os.path.join(original_data_path, image_name))
        image_data = image.img_to_array(image_data)
        image_data = tf.image.resize(image_data, [128, 128])  

        image.save_img(os.path.join(resize_data_path, image_name), image_data, data_format = 'channels_last')

def image_process(original_data_path, process_data_path, number_of_process, process_type):
    #取得資料夾中的檔名
    image_list = os.listdir(original_data_path)
    #分割檔名list
    sub_list_index = []
    for sub_index in range(number_of_process):
        if (sub_index + 1) * math.ceil(len(image_list) / number_of_process) < len(image_list):
            sub_list_index.append((sub_index + 1) * math.ceil(len(image_list) / number_of_process))
        else:
            sub_list_index.append(len(image_list))
        
    start_index = 0
    if process_type == "smooth":
        for end_index in sub_list_index:
            process_list.append(executor.submit(image_smooth, image_list[start_index:end_index],
                                                original_data_path, process_data_path))
    if process_type == "resize":
        for end_index in sub_list_index:
            process_list.append(executor.submit(image_resize, image_list[start_index:end_index],
                                                original_data_path, process_data_path))
        start_index = end_index
    for process in concurrent.futures.as_completed(process_list):
        print('Process complet')


if __name__ == '__main__':

    #try:
    #    #取得實體GPU數量
    #    gpus = tf.config.experimental.list_physical_devices('GPU')
    #    if gpus:
    #        for gpu in gpus:
    #            #將GPU記憶體使用率設為動態成長
    #            #有建立虛擬GPU時不可使用
    #            #tf.config.experimental.set_memory_growth(gpu, True)
    #            #建立虛擬GPU
    #            tf.config.experimental.set_virtual_device_configuration(
    #                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 5000)])
    #        #確認虛擬GPU資訊    
    #        #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    #        #for logical_gpu in logical_gpus:
    #        #    tf.config.experimental.set_memory_growth(logical_gpu, True)
    #        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    #except Exception as e:
    #    print(e)

    #原始資料位置
    original_data_path = 'E:/python program/GAN/reality_to_animation/original/'
    #改變大小後的位置
    smooth_data_path = 'E:/python program/GAN/reality_to_animation/smooth/'
    resize_data_path = 'E:/python program/GAN/reality_to_animation/resize/'
    #產生的資料位置
    generator_data_path = 'E:/python program/GAN/reality_to_animation/generator/'
    original_animation_data_path = os.path.join(original_data_path, 'animation')
    original_reality_data_path = os.path.join(original_data_path, 'reality')
    resize_animation_data_path = os.path.join(resize_data_path, 'train/animation')
    resize_reality_data_path = os.path.join(resize_data_path, 'train/reality')
    generator_train_data_path = os.path.join(generator_data_path, 'train')
    generator_test_data_path = os.path.join(generator_data_path, 'test')
    #模型路徑
    model_path = 'E:/python program/GAN/reality_to_animation/generator/model/'

    

    #多執行緒數量
    number_of_process = 2
    #註冊多執行緒管理器
    executor = concurrent.futures.ProcessPoolExecutor(max_workers = number_of_process)
    #保留宣告的執行緒以便取得結果
    process_list = []

    resize_data_exist = False
    try:
        #如果資料夾不存在則建立資料夾
        os.mkdir(resize_animation_data_path) 
        os.mkdir(resize_reality_data_path)
    except FileExistsError:
        resize_data_exist = True

    #如果是第一次執行程式才要跑
    if not resize_data_exist:
        image_process(original_train_data_path, resize_train_data_path, number_of_process, "resize")
        image_process(original_test_data_path, resize_test_data_path, number_of_process, "resize")

    def generator_model():
        start_imagesize = 128
        imagesize = 128
        nodes = 64
        max_nodes = 512
        max_count = 0
        stride = 2
        bn_layer_list = []
        keep = True

        input_layer   = Input(shape = (imagesize, imagesize, 3))
        con2d_layer   = layers.Conv2D(nodes, 3, stride, padding = 'same')(input_layer)
        hidden_layer  = layers.PReLU()(con2d_layer)
        bn_layer_list.append(layers.BatchNormalization()(hidden_layer))
        imagesize     = math.ceil(imagesize / stride)

        while keep:
            if nodes < max_nodes:
                nodes *= 2
            else:
                max_count += 1
            con2d_layer   = layers.Conv2D(nodes, 3, stride,padding = 'same')(bn_layer_list[-1])
            hidden_layer  = layers.PReLU()(con2d_layer)
            bn_layer_list.append(layers.BatchNormalization()(hidden_layer))
            
            imagesize     = math.ceil(imagesize / stride)
            if imagesize == 1:
                keep = False
         
        keep = True
   
        while keep:
            if imagesize != 1:
                input_data = layers.concatenate([bn_layer, bn_layer_list.pop()])
            else:
                input_data = bn_layer_list.pop()
            upsample_layer = layers.UpSampling2D((4,4))(input_data)
            con2d_layer    = layers.Conv2D(nodes, 3, stride,padding = 'same')(upsample_layer)
            hidden_layer   = layers.PReLU()(con2d_layer)
            bn_layer       = layers.BatchNormalization()(hidden_layer)
            imagesize      = math.ceil(imagesize * stride)
            if max_count != 1:
                max_count -= 1
            else:
                nodes /= 2
            if imagesize == (start_imagesize / 2):
                keep = False

        upsample_layer = layers.UpSampling2D((4,4))(bn_layer)
        output_layer   = layers.Conv2D(3, 3, stride,padding = 'same', activation = 'tanh')(upsample_layer)
        #output_layer = layers.Conv2DTranspose(3, 3, stride,padding = 'same', activation = 'tanh')(bn_layer)
        #con2d_layer   = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(bn_layer_1)
        #bn_layer_2    = layers.BatchNormalization()(con2d_layer)
        #add_layer_1   = layers.add([bn_layer_1, bn_layer_2])
        #con2d_layer   = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(add_layer_1)
        #bn_layer_3    = layers.BatchNormalization()(con2d_layer)
        #add_layer_2   = layers.add([add_layer_1, bn_layer_3])
        #con2d_layer   = layers.Conv2D(64, 3, padding = 'same', activation = 'relu')(add_layer_2)
        #bn_layer_4    = layers.BatchNormalization()(con2d_layer)
        #add_layer_3   = layers.add([add_layer_2, bn_layer_4])
        #con2d_layer   = layers.Conv2D(3, 3, padding = 'same', activation = 'tanh')(add_layer_3)
        #output_layer  = layers.maximum([input_layer, con2d_layer])

        return Model(input_layer, output_layer)

    def discriminator_model(window_size, layer_name_list):
        stride = 2
        input_layer   = Input(shape = (window_size, window_size, 3, ))
        #con2d_layer   = layers.Conv2D(64, 3, stride,padding = 'same')(input_layer)
        #act_layer     = layers.PReLU()(con2d_layer)
        #bn_layer      = layers.BatchNormalization()(act_layer)
        #con2d_layer   = layers.Conv2D(128, 3, stride,padding = 'same')(bn_layer)
        #act_layer     = layers.PReLU()(con2d_layer)
        #bn_layer      = layers.BatchNormalization()(act_layer)
        #con2d_layer   = layers.Conv2D(256, 3, stride,padding = 'same')(bn_layer)
        #act_layer     = layers.PReLU()(con2d_layer)
        #bn_layer      = layers.BatchNormalization()(act_layer)
        #con2d_layer   = layers.Conv2D(512, 3, stride,padding = 'same')(bn_layer)
        #act_layer     = layers.PReLU()(con2d_layer)
        #bn_layer      = layers.BatchNormalization()(act_layer)
        #output_layer  = layers.Conv2D(1, 3, 1, padding = 'same')(bn_layer)
        con2d_layer   = layers.Conv2D(16, 3, name = layer_name_list[0])(input_layer)
        act_layer     = layers.PReLU(name = layer_name_list[1])(con2d_layer)
        bn_layer      = layers.BatchNormalization(name = layer_name_list[2])(act_layer)
        dropout_layer = layers.Dropout(0.5, name = layer_name_list[3])(bn_layer)
        con2d_layer   = layers.Conv2D(32, 3, name = layer_name_list[4])(dropout_layer)
        act_layer     = layers.PReLU(name = layer_name_list[5])(con2d_layer)
        bn_layer      = layers.BatchNormalization(name = layer_name_list[6])(act_layer)
        dropout_layer = layers.Dropout(0.5, name = layer_name_list[7])(bn_layer)
        con2d_layer   = layers.Conv2D(64, 3, name = layer_name_list[8])(dropout_layer)
        act_layer     = layers.PReLU(name = layer_name_list[9])(con2d_layer)
        bn_layer      = layers.BatchNormalization(name = layer_name_list[10])(act_layer)
        dropout_layer = layers.Dropout(0.5, name = layer_name_list[11])(bn_layer)
        flatten_layer = layers.Flatten()(dropout_layer)
        hidden_layer  = layers.Dense(20)(flatten_layer)
        act_layer     = layers.PReLU()(hidden_layer)
        dropout_layer = layers.Dropout(0.5)(act_layer)
        hidden_layer  = layers.Dense(40)(dropout_layer)
        act_layer     = layers.PReLU()(hidden_layer)
        dropout_layer = layers.Dropout(0.5)(act_layer)
        output_layer  = layers.Dense(1, activation = 'sigmoid')(dropout_layer)

        return Model(input_layer, output_layer)

    def classification_model(window_size, layer_name_list):
        input_layer   = Input(shape = (window_size, window_size, 3, ))
        con2d_layer   = layers.Conv2D(16, 3, name = layer_name_list[0])(input_layer)
        act_layer     = layers.PReLU(name = layer_name_list[1])(con2d_layer)
        bn_layer      = layers.BatchNormalization(name = layer_name_list[2])(act_layer)
        dropout_layer = layers.Dropout(0.5, name = layer_name_list[3])(bn_layer)
        con2d_layer   = layers.Conv2D(32, 3, name = layer_name_list[4])(dropout_layer)
        act_layer     = layers.PReLU(name = layer_name_list[5])(con2d_layer)
        bn_layer      = layers.BatchNormalization(name = layer_name_list[6])(act_layer)
        dropout_layer = layers.Dropout(0.5, name = layer_name_list[7])(bn_layer)
        con2d_layer   = layers.Conv2D(64, 3, name = layer_name_list[8])(dropout_layer)
        act_layer     = layers.PReLU(name = layer_name_list[9])(con2d_layer)
        bn_layer      = layers.BatchNormalization(name = layer_name_list[10])(act_layer)
        dropout_layer = layers.Dropout(0.5, name = layer_name_list[11])(bn_layer)
        flatten_layer = layers.Flatten()(dropout_layer)
        hidden_layer  = layers.Dense(20)(flatten_layer)
        act_layer     = layers.PReLU()(hidden_layer)
        dropout_layer = layers.Dropout(0.5)(act_layer)
        hidden_layer  = layers.Dense(40)(dropout_layer)
        act_layer     = layers.PReLU()(hidden_layer)
        dropout_layer = layers.Dropout(0.5)(act_layer)
        output_layer  = layers.Dense(1, activation = 'sigmoid')(dropout_layer)

        return Model(input_layer, output_layer)

    @tf.function
    def gmodel_loss(gmodel_reality_to_Animation, gmodel_reality_to_Animation_optimizer,
                                         gmodel_Animation_to_reality, gmodel_Animation_to_reality_optimizer,
                                         dmodel_Animation, dmodel_reality, batch_reality_image, batch_Animation_image):
        with tf.GradientTape() as reality_to_Animation_tape:
            #biase = tf.convert_to_tensor(1e-20)
            fake_Animation_image    = gmodel_reality_to_Animation(batch_reality_image)
            fake_Animation_predict  = dmodel_Animation(fake_Animation_image)
            fake_reality_image      = gmodel_Animation_to_reality(batch_Animation_image)
            recover_reality_image   = gmodel_Animation_to_reality(fake_Animation_image)
            recover_Animation_image = gmodel_reality_to_Animation(fake_reality_image)
            #確保不會出現log0
            #fake_Animation_predict  = tf.math.add(fake_Animation_predict, biase)
            #tensorflow 的優化器不能處理負值 loss value
            #loss_value  = tf.math.reduce_mean(tf.math.abs(tf.math.log(fake_Animation_predict)))
            #loss_value += (tf.math.reduce_mean(tf.keras.losses.mse(batch_reality_image, recover_reality_image)) * 10)
            reality_to_Animation_loss_value  = tf.math.reduce_mean(tf.math.squared_difference(fake_Animation_predict, 1))
            cyclic_loss_value = tf.math.add(tf.math.reduce_mean(tf.keras.losses.mae(batch_Animation_image, recover_Animation_image)), 
                                            tf.math.reduce_mean(tf.keras.losses.mae(batch_reality_image, fake_reality_image)))
            reality_to_Animation_loss_value = (reality_to_Animation_loss_value) + (cyclic_loss_value * 5)

        #計算梯度
        reality_to_Animation_weight    = gmodel_reality_to_Animation.trainable_weights
        reality_to_Animation_gradients = reality_to_Animation_tape.gradient(reality_to_Animation_loss_value, reality_to_Animation_weight)
        
        with tf.GradientTape() as Animation_to_reality_tape:
            #biase = tf.convert_to_tensor(1e-20)
            fake_reality_image      = gmodel_Animation_to_reality(batch_Animation_image)
            fake_reality_predict    = dmodel_reality(fake_reality_image)
            fake_Animation_image    = gmodel_reality_to_Animation(batch_reality_image)
            recover_reality_image   = gmodel_Animation_to_reality(fake_Animation_image)
            recover_Animation_image = gmodel_reality_to_Animation(fake_reality_image)
            #確保不會出現log0
            #fake_Animation_predict  = tf.math.add(fake_Animation_predict, biase)
            #tensorflow 的優化器不能處理負值 loss value
            #loss_value  = tf.math.reduce_mean(tf.math.abs(tf.math.log(fake_Animation_predict)))
            #loss_value += (tf.math.reduce_mean(tf.keras.losses.mse(batch_reality_image, recover_reality_image)) * 10)
            Animation_to_reality_loss_value  = tf.math.reduce_mean(tf.math.squared_difference(fake_reality_predict, 1))
            cyclic_loss_value = tf.math.add(tf.math.reduce_mean(tf.keras.losses.mae(batch_Animation_image, recover_Animation_image)),
                                            tf.math.reduce_mean(tf.keras.losses.mae(batch_reality_image, fake_reality_image)))
            Animation_to_reality_loss_value = (Animation_to_reality_loss_value) + (cyclic_loss_value * 5)

        #計算梯度
        Animation_to_reality_weight = gmodel_Animation_to_reality.trainable_weights
        Animation_to_reality_gradients = Animation_to_reality_tape.gradient(Animation_to_reality_loss_value, Animation_to_reality_weight)

        #根據梯度更新權重
        gmodel_reality_to_Animation_optimizer.apply_gradients(zip(reality_to_Animation_gradients, reality_to_Animation_weight))
        gmodel_Animation_to_reality_optimizer.apply_gradients(zip(Animation_to_reality_gradients, Animation_to_reality_weight))

        return reality_to_Animation_loss_value, Animation_to_reality_loss_value
    @tf.function
    def gmodel_Animation_to_reality_loss(gmodel_Animation_to_reality, gmodel_Animation_to_reality_optimizer,
                                         gmodel_reality_to_Animation, dmodel_reality, batch_Animation_image):
        with tf.GradientTape() as tape:
            biase = tf.convert_to_tensor(1e-20)
            fake_reality_image       = gmodel_Animation_to_reality(batch_Animation_image)
            recover_Animation_image  = gmodel_reality_to_Animation(fake_reality_image)
            fake_reality_predict     = dmodel_reality(fake_reality_image)
            #確保不會出現log0
            fake_reality_predict  = tf.math.add(fake_reality_predict, biase)
            #tensorflow 的優化器不能處理負值 loss value
            loss_value  = tf.math.reduce_mean(tf.math.abs(tf.math.log(fake_reality_predict)))
            loss_value += (tf.math.reduce_mean(tf.keras.losses.mse(batch_Animation_image, recover_Animation_image)) * 10)
        #計算梯度
        weight = gmodel_Animation_to_reality.trainable_weights
        gradients = tape.gradient(loss_value, weight)
        #根據梯度更新權重
        gmodel_Animation_to_reality_optimizer.apply_gradients(zip(gradients, weight))


        return loss_value

    @tf.function
    def dmodel_Animation_loss(dmodel_Animation, dmodel_Animation_optimizer, gmodel_reality_to_Animation, 
                              batch_reality_image, batch_Animation_image):
        with tf.GradientTape() as tape:
            Animation_predict      = dmodel_Animation(batch_Animation_image)
            fake_Animation_image   = gmodel_reality_to_Animation(batch_reality_image)
            fake_Animation_predict = dmodel_Animation(fake_Animation_image)
            #tensorflow 的優化器不能處理負值 loss value
            #loss_value  = tf.math.reduce_mean(tf.math.abs(tf.math.log(Animation_predict)) +
            #                                  tf.math.abs(tf.math.log(1 - fake_Animation_predict)))
            loss_value  = tf.math.add(tf.math.reduce_mean(tf.math.squared_difference(Animation_predict, 1)),
                                      tf.math.reduce_mean(tf.math.square(fake_Animation_predict)))
            loss_value  = tf.math.divide(loss_value, 2)

        #計算梯度
        weight = dmodel_Animation.trainable_weights
        gradients = tape.gradient(loss_value, weight)
        #根據梯度更新權重
        dmodel_Animation_optimizer.apply_gradients(zip(gradients, weight))
        return loss_value

    @tf.function
    def dmodel_reality_loss(dmodel_reality, dmodel_reality_optimizer, gmodel_Animation_to_reality, 
                            batch_reality_image, batch_Animation_image):
        with tf.GradientTape() as tape:
            reality_predict      = dmodel_reality(batch_reality_image)
            fake_reality_image   = gmodel_Animation_to_reality(batch_Animation_image)
            fake_reality_predict = dmodel_reality(fake_reality_image)
            #tensorflow 的優化器不能處理負值 loss value
            #loss_value  = tf.math.reduce_mean(tf.math.abs(tf.math.log(reality_predict)) +
            #                                  tf.math.abs(tf.math.log(1 - fake_reality_predict)))
            loss_value  = tf.math.add(tf.math.reduce_mean(tf.math.squared_difference(reality_predict, 1)),
                                      tf.math.reduce_mean(tf.math.square(fake_reality_predict)))
            loss_value  = tf.math.divide(loss_value, 2)

        #計算梯度
        weight = dmodel_reality.trainable_weights
        gradients = tape.gradient(loss_value, weight)
        #根據梯度更新權重
        dmodel_reality_optimizer.apply_gradients(zip(gradients, weight))
        return loss_value

    def model_training(gmodel_reality_to_Animation, gmodel_reality_to_Animation_optimizer,
                       gmodel_Animation_to_reality, gmodel_Animation_to_reality_optimizer,
                       dmodel_reality, dmodel_reality_optimizer, dmodel_Animation, dmodel_Animation_optimizer,
                       batch_reality_image, batch_Animation_image, first_training, train_gmodel):
        #在GradientTape中計算loss_value以便計算梯度
        if first_training:
            reality_to_Animation_loss, Animation_to_reality_loss = gmodel_loss(gmodel_reality_to_Animation, gmodel_reality_to_Animation_optimizer,
                                                                               gmodel_Animation_to_reality, gmodel_Animation_to_reality_optimizer,
                                                                               dmodel_Animation, dmodel_reality, batch_reality_image, batch_Animation_image)
            #Animation_to_reality_loss = gmodel_Animation_to_reality_loss(gmodel_Animation_to_reality, gmodel_Animation_to_reality_optimizer,
            #                                                             gmodel_reality_to_Animation, dmodel_reality, batch_reality_image,
            #                                                             batch_Animation_image)
            Animation_loss = dmodel_Animation_loss(dmodel_Animation, dmodel_Animation_optimizer, gmodel_reality_to_Animation, 
                                                   batch_reality_image, batch_Animation_image)
            reality_loss   = dmodel_reality_loss(dmodel_reality, dmodel_reality_optimizer, gmodel_Animation_to_reality,
                                                 batch_reality_image, batch_Animation_image)
            return [reality_to_Animation_loss, Animation_to_reality_loss, Animation_loss, reality_loss]
        else:
            if train_gmodel:
                reality_to_Animation_loss, Animation_to_reality_loss = gmodel_loss(gmodel_reality_to_Animation, gmodel_reality_to_Animation_optimizer,
                                                                                   gmodel_Animation_to_reality, gmodel_Animation_to_reality_optimizer,
                                                                                   dmodel_Animation, dmodel_reality, batch_reality_image, batch_Animation_image)
                #Animation_to_reality_loss = gmodel_Animation_to_reality_loss(gmodel_Animation_to_reality, gmodel_Animation_to_reality_optimizer,
                #                                                         gmodel_reality_to_Animation, dmodel_reality, batch_reality_image, 
                #                                                         batch_Animation_image)
                Animation_loss = 100
                reality_loss   = 100
            else:
                Animation_loss = dmodel_Animation_loss(dmodel_Animation, dmodel_Animation_optimizer, gmodel_reality_to_Animation, 
                                                   batch_reality_image, batch_Animation_image)
                reality_loss   = dmodel_reality_loss(dmodel_reality, dmodel_reality_optimizer, gmodel_Animation_to_reality, 
                                                     batch_reality_image, batch_Animation_image)
                reality_to_Animation_loss = 100
                Animation_to_reality_loss = 100

            return [reality_to_Animation_loss, Animation_to_reality_loss, Animation_loss, reality_loss]     
    
    #驗證產生器的結果
    def generator_image(gmodel, input_data, generator_data_path, image_label, epoch):
        predict = gmodel.predict(input_data)
        #predict = Binarization(predict, gate_value)
        #匯出資料
        for output_count in range(predict.shape[0]):
            image_name = "epoch_" + str(epoch) + ' image_label_' + image_label + str(output_count) + ".png"
            image.save_img(os.path.join(generator_data_path, image_name),
                           np.concatenate((predict[output_count, :, :, :], input_data.numpy()[output_count, :, :, :]), axis = 1),
                           data_format = 'channels_last')
            #image_count += 1
                
    #取的檔案類別
    def get_file_label(file_path):
        #以python內建的檔案路徑分隔符號為標準分割字串
        split_string = tf.strings.split(file_path, os.path.sep)
        #從檔案名稱分割出類別的部分
        #file_label = tf.strings.split(split_string[-2], '.')[0]
        #回傳類別
        return split_string[-2]

    #將image傳成tensor
    def decode_image(image):
        #將image轉成nuit8 channel為3的tensor
        image = tf.io.decode_image(image, channels = 3)
        #將image轉成指定的data type，並且作正規化
        image = tf.image.convert_image_dtype(image, tf.float32)
        #image = tf.cast(image, tf.float32)
        return image

    #將影像分割成訓練和測試影像
    def image_split(train_image):
        if len(tf.shape(train_image)) < 4:
            train_image = tf.expand_dims(train_image, 0)
        
        train_image, test_image = tf.split(train_image, num_or_size_splits = 2, axis = -2)    

        return train_image, test_image

    #路徑處理function
    def process_path(file_path):
        file_label = get_file_label(file_path)
        #匯入檔案
        image = tf.io.read_file(file_path)
        image = decode_image(image)
        return file_label, image

    #超參數設定
    #是否為第一次訓練
    first_training = True
    #辨識器輸入影像大小
    window_size = 128
    #辨識器捲積層名稱
    layer_name_list = ['my_con2d_1', 'my_prelu_1', 'my_bn_1', 'my_dropout_1',
                       'my_con2d_2', 'my_prelu_2', 'my_bn_2', 'my_dropout_2',
                       'my_con2d_3', 'my_prelu_3', 'my_bn_3', 'my_dropout_3']
    #建立產生器模型
    gmodel_reality_to_Animation = generator_model()
    gmodel_reality_to_Animation.summary()
    gmodel_Animation_to_reality = generator_model()
    #建立辨識器模型
    dmodel_reality = discriminator_model(window_size, layer_name_list)
    dmodel_reality.summary()
    dmodel_Animation = discriminator_model(window_size, layer_name_list)
    #建立分類器模型
    cmodel = classification_model(window_size, layer_name_list)
    #設定最佳化方式
    if first_training:
        learning_rate = 1e-4
        gmodel_reality_to_Animation_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        gmodel_Animation_to_reality_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        dmodel_reality_optimizer   = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        dmodel_Animation_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
    else:
        learning_rate = 1e-4
        gmodel_reality_to_Animation_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        gmodel_Animation_to_reality_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        dmodel_reality_optimizer   = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        dmodel_Animation_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)

    training_classification = False
    if training_classification:
        cmodel_optimizer = tf.keras.optimizers.Adam()

    #批次數量
    batch_size = 1
    image_count = 1
    if first_training:
        epochs = 5000
    else:
        epochs = 200
    
    #匯入資料
    if training_classification:
        train_image_list = Dataset.list_files(resize_data_path + "*.jpg")
        train_image_dataset = train_image_list.map(process_path, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        train_image_dataset = train_image_dataset.shuffle(62).repeat()
        train_image_dataset = train_image_dataset.batch(batch_size)
        steps_each_epoch = math.ceil(62 / batch_size)
    else:
        reality_image_list = Dataset.list_files(resize_reality_data_path + "/*.jpg")
        reality_image_dataset = reality_image_list.map(process_path, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        reality_image_dataset = reality_image_dataset.shuffle(38).repeat()
        reality_image_dataset = reality_image_dataset.batch(batch_size)

        Animation_image_list = Dataset.list_files(resize_animation_data_path + "/*.jpg")
        Animation_image_dataset = Animation_image_list.map(process_path, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        Animation_image_dataset = Animation_image_dataset.shuffle(38).repeat()
        Animation_image_dataset = Animation_image_dataset.batch(batch_size)

        steps_each_epoch = math.ceil(18 / batch_size)
        
    #GAN訓練主體
    def train_GAN(gmodel_reality_to_Animation, gmodel_reality_to_Animation_optimizer,
                  gmodel_Animation_to_reality, gmodel_Animation_to_reality_optimizer,
                  dmodel_reality, dmodel_reality_optimizer, dmodel_Animation, dmodel_Animation_optimizer,
                  reality_image_dataset, Animation_image_dataset, batch_size, epochs, steps_each_epoch,
                  model_path, first_training, generator_train_data_path, layer_name_list, learning_rate):
        train_gmodel = False
        if first_training:
            #gmodel_reality_to_Animation = tf.keras.models.load_model(model_path + "first_gmodel_reality_to_Animation")
            #gmodel_Animation_to_reality = tf.keras.models.load_model(model_path + "first_gmodel_Animation_to_reality")
            #dmodel_reality   = tf.keras.models.load_model(model_path + "first_dmodel_reality")
            #dmodel_Animation = tf.keras.models.load_model(model_path + "first_dmodel_Animation")
            cmodel = tf.keras.models.load_model(model_path + "c_model")
            #weight_copy(dmodel_reality, cmodel, layer_name_list)
            #weight_copy(dmodel_Animation, cmodel, layer_name_list)
        else:
            gmodel_reality_to_Animation = tf.keras.models.load_model(model_path + "first_gmodel_reality_to_Animation")
            gmodel_Animation_to_reality = tf.keras.models.load_model(model_path + "first_gmodel_Animation_to_reality")
            dmodel_reality   = tf.keras.models.load_model(model_path + "first_dmodel_reality")
            dmodel_Animation = tf.keras.models.load_model(model_path + "first_dmodel_Animation")
            
            gmodel_reality_to_Animation.trainable = True
            gmodel_Animation_to_reality.trainable = True
            dmodel_reality.trainable = False
            dmodel_Animation.trainable = False
            #if train_gmodel:
            #    gmodel_reality_to_Animation.trainable = True
            #    gmodel_Animation_to_reality.trainable = True
            #    dmodel_reality.trainable = False
            #    dmodel_Animation.trainable = False
            #else:
            #    gmodel.trainable = False
            #    dmodel.trainable = True

        reality_image_dataset_iterator = iter(reality_image_dataset)
        Animation_image_dataset_iterator = iter(Animation_image_dataset)
        for epoch in range(epochs):
            g_reality_loss_list   = []
            g_Animation_loss_list = []
            d_reality_loss_list   = []
            d_Animation_loss_list = []
            step = 0
            print('Epoch:' + str(epoch))

            while step < steps_each_epoch:
                start_time = time.time()
                #從dataset batch匯入檔案
                batch_label, batch_reality_image = next(reality_image_dataset_iterator)
                batch_label, batch_Animation_image = next(Animation_image_dataset_iterator)
                #batch_glabel = label_process(batch_label, glabel)
                #batch_dlabel = label_process(batch_label, dlabel)
                output_list = model_training(gmodel_reality_to_Animation, gmodel_reality_to_Animation_optimizer,
                                             gmodel_Animation_to_reality, gmodel_Animation_to_reality_optimizer,
                                             dmodel_reality, dmodel_reality_optimizer, dmodel_Animation,
                                             dmodel_Animation_optimizer, batch_reality_image, batch_Animation_image,
                                             first_training, train_gmodel)
            
                g_Animation_loss_list.append(output_list[0])
                g_reality_loss_list.append(output_list[1])
                d_Animation_loss_list.append(output_list[2])
                d_reality_loss_list.append(output_list[3])


                step += 1
                end_time = time.time()
                ETA_time(start_time, end_time, steps_each_epoch, step)
            
            print('G reality Model Loss:' + str(np.mean(g_reality_loss_list)))
            print('G Animation Model Loss:' + str(np.mean(g_Animation_loss_list)))
            print('D reality Model Loss:' + str(np.mean(d_reality_loss_list)))
            print('D Animation Model Loss:' + str(np.mean(d_Animation_loss_list)))

            if first_training:
                generator_image(gmodel_reality_to_Animation, batch_reality_image, generator_train_data_path, 'r_to_a', epoch)
                generator_image(gmodel_Animation_to_reality, batch_Animation_image, generator_train_data_path, 'a_to_r', epoch)
                if epoch % 100 == 0:
                    if epoch > 0:
                        gmodel_reality_to_Animation.save(model_path + "first_gmodel_reality_to_Animation")
                        gmodel_Animation_to_reality.save(model_path + "first_gmodel_Animation_to_reality")
                        dmodel_reality.save(model_path + "first_dmodel_reality")
                        dmodel_Animation.save(model_path + "first_dmodel_Animation")
                        learning_rate = learning_rate / 2
                        gmodel_reality_to_Animation_optimizer.learning_rate = learning_rate
                        gmodel_Animation_to_reality_optimizer.learning_rate = learning_rate
                        dmodel_reality_optimizer.learning_rate = learning_rate
                        dmodel_Animation_optimizer.learning_rate = learning_rate
            else:
                if train_gmodel:
                    #驗證g model目前能產生的結果
                    generator_image(gmodel_reality_to_Animation, batch_reality_image, generator_train_data_path, 'r_to_a', epoch)
                    generator_image(gmodel_Animation_to_reality, batch_Animation_image, generator_train_data_path, 'a_to_r', epoch)
                    if np.mean(np.mean(g_reality_loss_list) + np.mean(g_Animation_loss_list)) < 0.06:
                        train_gmodel = False
                        gmodel_reality_to_Animation.trainable = False
                        gmodel_Animation_to_reality.trainable = False
                        dmodel_reality.trainable = True
                        dmodel_Animation.trainable = True
                else:
                    if np.mean(np.mean(d_reality_loss_list) + np.mean(d_Animation_loss_list)) < 0.001:
                        train_gmodel = True
                        gmodel_reality_to_Animation.trainable = True
                        gmodel_Animation_to_reality.trainable = True
                        dmodel_reality.trainable = False
                        dmodel_Animation.trainable = False
                
                if epoch % 100 == 0:
                    gmodel_reality_to_Animation.save(model_path + "second_gmodel_reality_to_Animation")
                    gmodel_Animation_to_reality.save(model_path + "second_gmodel_Animation_to_reality")
                    dmodel_reality.save(model_path + "second_dmodel_reality")
                    dmodel_Animation.save(model_path + "second_dmodel_Animation")

    #權重複製
    def weight_copy(target_model, weight_model, layer_name_list):
        for layer_name in layer_name_list:
            tlayer = target_model.get_layer(name = layer_name)
            wlayer = weight_model.get_layer(name = layer_name)
            tlayer.set_weights(wlayer.get_weights())
            tlayer.trainable = False

    #分類類別數值化
    def label_process(batch_label, label_list):
        output_label = []
        batch_label = batch_label.numpy()
        for i in range(len(batch_label)):
            if bytes.decode(batch_label[i]) == label_list[0]:
                output_label.append(0)
            else:
                output_label.append(1)
        output_label = tf.convert_to_tensor(output_label, tf.float32)
        output_label = tf.expand_dims(output_label, -1)
        return output_label

    @tf.function
    def classification_loss(cmodel, cmodel_optimizer, image, label):
        with tf.GradientTape() as tape:
            predict = cmodel(image)
            loss = tf.keras.losses.binary_crossentropy(label, predict)
        #計算梯度
        weight = cmodel.trainable_weights
        gradients = tape.gradient(loss, weight)
        #根據梯度更新權重
        cmodel_optimizer.apply_gradients(zip(gradients, weight))
        return loss

    #預計剩餘時間
    def ETA_time(start_time, end_time, steps_each_epoch, step):     
        run_time = end_time -  start_time
        eta_time_seconds = math.floor(run_time * (steps_each_epoch - step))
        if eta_time_seconds > 3600:
            eta_time_hours = math.floor(eta_time_seconds / 3600)
            eta_time_seconds = eta_time_seconds - (3600 * eta_time_hours)
            eta_time_minutes = math.floor(eta_time_seconds / 60)
            eta_time_seconds = eta_time_seconds - (60 * eta_time_minutes)
            output_str = f"預計剩餘時間: {eta_time_hours}小時 {eta_time_minutes}分鐘 {eta_time_seconds}秒"
        elif eta_time_seconds > 60:
            eta_time_minutes = math.floor(eta_time_seconds / 60)
            eta_time_seconds = eta_time_seconds - (60 * eta_time_minutes)
            output_str = f"預計剩餘時間: {eta_time_minutes}分鐘 {eta_time_seconds}秒"
        else:
            output_str = f"預計剩餘時間: {eta_time_seconds}秒"
            
        lenght = len(output_str) * 2
        print(output_str, end = '\r')

    #辨識器預先訓練
    def train_classification(cmodel, cmodel_optimizer, train_image_dataset, batch_size, epochs, steps_each_epoch, model_path):

        label = ['reality', 'animation']
        print()
        train_image_dataset_iterator = iter(train_image_dataset)
        for epoch in range(epochs):
            step = 0
            loss_list = []
            while step < steps_each_epoch:
                start_time = time.time()
                batch_label, batch_image = next(train_image_dataset_iterator)
                batch_label = label_process(batch_label, label)
                loss_list.append(classification_loss(cmodel, cmodel_optimizer, batch_image, batch_label))
                step += 1
                end_time = time.time()
                ETA_time(start_time, end_time, steps_each_epoch, step)
            print("loss: " + str(np.mean(loss_list)))
            print()
        cmodel.save(model_path + "c_model")
    
    def acc_test(resize_train_data_path, model_path, batch_size):
        train_image_list = Dataset.list_files(resize_train_data_path + "*.jpg")
        train_image_dataset = train_image_list.map(process_path, num_parallel_calls = tf.data.experimental.AUTOTUNE)
        train_image_dataset = train_image_dataset.shuffle(62)
        train_image_dataset = train_image_dataset.batch(batch_size)

        steps_each_epoch = math.ceil(62 / batch_size)
        cmodel = tf.keras.models.load_model(model_path + "c_model")

        step = 0
        acc_list = []
        label = ['reality', 'animation']
        train_image_dataset_iterator = iter(train_image_dataset)
        while step < steps_each_epoch:
            batch_label, batch_image = next(train_image_dataset_iterator)
            batch_label = label_process(batch_label, label)
            predict = tf.round(cmodel.predict(batch_image))
            predict = tf.math.equal(batch_label, predict)
            acc_list.append(tf.reduce_mean(tf.cast(predict, tf.float32)))
            step += 1
        acc_mean = np.mean(acc_list)
        print(f"Acc: {acc_mean}")

    if training_classification:
        train_classification(cmodel, cmodel_optimizer, train_image_dataset, batch_size, epochs, steps_each_epoch, model_path)
        acc_test(resize_data_path, model_path, batch_size)
    else:
        train_GAN(gmodel_reality_to_Animation, gmodel_reality_to_Animation_optimizer,
                  gmodel_Animation_to_reality, gmodel_Animation_to_reality_optimizer,
                  dmodel_reality, dmodel_reality_optimizer, dmodel_Animation, dmodel_Animation_optimizer,
                  reality_image_dataset, Animation_image_dataset, batch_size, epochs, steps_each_epoch,
                  model_path, first_training, generator_train_data_path, layer_name_list, learning_rate)



        
            
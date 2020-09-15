import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras import Model
from tensorflow.keras.preprocessing import image
from tensorflow.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import os
import concurrent.futures


if __name__ == '__main__':

    try:
        #取得實體GPU數量
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                #將GPU記憶體使用率設為動態成長
                #有建立虛擬GPU時不可使用
                #tf.config.experimental.set_memory_growth(gpu, True)
                #建立虛擬GPU
                tf.config.experimental.set_virtual_device_configuration(
                    gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 2000)])
            #確認虛擬GPU資訊    
            #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            #for logical_gpu in logical_gpus:
            #    tf.config.experimental.set_memory_growth(logical_gpu, True)
            #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except Exception as e:
        print(e)

    #產生的資料位置
    generator_data_path = 'E:/python program/GAN/dogs-vs-cats/generator/'
    generator_train_data_path = os.path.join(generator_data_path, 'train')
    generator_test_data_path = os.path.join(generator_data_path, 'test')
    #模型路徑
    model_path = 'E:/python program/GAN/dogs-vs-cats/generator/model/'

    #生成器模型
    def generator_model():
        input_layer   = Input(shape = (10, 1, 1,))
        hidden_layer  = layers.Dense(128)(input_layer)
        hidden_layer  = layers.PReLU()(hidden_layer)
        con2dt_layer  = layers.Conv2DTranspose(64, [7, 10])(hidden_layer)
        act_layer     = layers.PReLU()(con2dt_layer)
        bn_layer      = layers.BatchNormalization()(act_layer)
        con2dt_layer  = layers.Conv2DTranspose(32, [7, 10])(bn_layer)
        act_layer     = layers.PReLU()(con2dt_layer)
        bn_layer      = layers.BatchNormalization()(act_layer)
        output_layer  = layers.Conv2DTranspose(1, [7, 10], activation = 'tanh')(bn_layer)
        
        return Model(input_layer, output_layer)
    #辨識器模型
    def discriminator_model(window_size, layer_name_list):
        input_layer   = Input(shape = (window_size, window_size, 1, ))
        con2d_layer   = layers.Conv2D(8, 2, name = layer_name_list[0])(input_layer)
        act_layer     = layers.PReLU(name = layer_name_list[1])(con2d_layer)
        bn_layer      = layers.BatchNormalization(name = layer_name_list[2])(act_layer)
        dropout_layer = layers.Dropout(0.5, name = layer_name_list[3])(bn_layer)
        con2d_layer   = layers.Conv2D(16, 2, name = layer_name_list[4])(dropout_layer)
        act_layer     = layers.PReLU(name = layer_name_list[5])(con2d_layer)
        bn_layer      = layers.BatchNormalization(name = layer_name_list[6])(act_layer)
        dropout_layer = layers.Dropout(0.5, name = layer_name_list[7])(bn_layer)
        con2d_layer   = layers.Conv2D(32, 2, name = layer_name_list[8])(dropout_layer)
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
    #目標影像分類器模型
    def classification_model(window_size, layer_name_list):
        input_layer   = Input(shape = (window_size, window_size, 1, ))
        con2d_layer   = layers.Conv2D(8, 2, name = layer_name_list[0])(input_layer)
        act_layer     = layers.PReLU(name = layer_name_list[1])(con2d_layer)
        bn_layer      = layers.BatchNormalization(name = layer_name_list[2])(act_layer)
        dropout_layer = layers.Dropout(0.5, name = layer_name_list[3])(bn_layer)
        con2d_layer   = layers.Conv2D(16, 2, name = layer_name_list[4])(dropout_layer)
        act_layer     = layers.PReLU(name = layer_name_list[5])(con2d_layer)
        bn_layer      = layers.BatchNormalization(name = layer_name_list[6])(act_layer)
        dropout_layer = layers.Dropout(0.5, name = layer_name_list[7])(bn_layer)
        con2d_layer   = layers.Conv2D(32, 2, name = layer_name_list[8])(dropout_layer)
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
        output_layer  = layers.Dense(10, activation = 'softmax')(dropout_layer)

        return Model(input_layer, output_layer)

    #影像二值化
    def Binarization(image, gate_value):
        gate_value = tf.constant(gate_value)
        binary_mask = tf.math.greater(image, gate_value)
        binary_image = tf.where(binary_mask, image, 0)
        return binary_image

    #生成器訓練
    @tf.function
    def g_model_loss(gmodel, dmodel, gmodel_optimizer, train_data_list, original_data_list):
        with tf.GradientTape() as tape:
            fake_data = gmodel(train_data_list)
            fake_predict = dmodel(fake_data)
            #tensorflow 的優化器不能處理負值 loss value
            loss_value = tf.math.reduce_mean(tf.math.abs(tf.math.log(fake_predict))) * 1.2
            loss_value += tf.math.reduce_mean(tf.keras.losses.mse(original_data_list, fake_data))
        #計算梯度
        weight = gmodel.trainable_weights
        gradients = tape.gradient(loss_value, weight)
        #根據梯度更新權重
        gmodel_optimizer.apply_gradients(zip(gradients, weight))

        return loss_value
    #辨識器訓練
    @tf.function
    def d_model_loss(gmodel, dmodel, dmodel_optimizer, train_data_list, original_data_list):
        with tf.GradientTape() as tape:
            true_predict = dmodel(original_data_list)
            fake_data = gmodel(train_data_list)
            fake_predict = dmodel(fake_data)
            #tensorflow 的優化器不能處理負值 loss value
            loss_value = tf.math.reduce_mean(tf.math.abs(tf.math.log(true_predict))
                                             + tf.math.abs(tf.math.log(1 - fake_predict)))
        #計算梯度
        weight = dmodel.trainable_weights
        gradients = tape.gradient(loss_value, weight)
        #根據梯度更新權重
        dmodel_optimizer.apply_gradients(zip(gradients, weight))
        return loss_value
    #模型訓練函式
    def model_training(gmodel, dmodel, gmodel_optimizer, dmodel_optimizer,
                       train_data_list, original_data_list, first_training, train_gmodel):
        #第一次訓練時辨識器的捲基層不練
        #以加速生成模型掌握辨識器的判斷標準
        if first_training:
            g_loss = g_model_loss(gmodel, dmodel, gmodel_optimizer, train_data_list, original_data_list)
            d_loss = d_model_loss(gmodel, dmodel, dmodel_optimizer, train_data_list, original_data_list)
            return g_loss, d_loss
        else:
            #第二次訓練時，一次只訓練一個模型以保持穩定性
            if train_gmodel:
                g_loss = g_model_loss(gmodel, dmodel, gmodel_optimizer, train_data_list, original_data_list)
                d_loss = 100
            else:
                d_loss = d_model_loss(gmodel, dmodel, dmodel_optimizer, train_data_list, original_data_list)
                g_loss = 100
            return g_loss, d_loss        
    
    #驗證產生器的結果
    def generator_image(gmodel, input_data, true_image, generator_data_path, image_count, epoch, gate_value):
        predict = gmodel.predict(input_data)
        predict = Binarization(predict, gate_value)
        #匯出資料
        for output_count in range(predict.shape[0]):
            image_name = "epoch_" + str(epoch) + ' image_label_' + str(image_count[output_count]) + ".png"
            image.save_img(os.path.join(generator_data_path, image_name),
                           np.concatenate((predict[output_count, :, :, :], true_image.numpy()[output_count, :, :, :]), axis = 1),
                           data_format = 'channels_last')
           
    #超參數設定
    #是否為第一次訓練
    first_training = False
    #辨識器輸入影像大小
    window_size = 28
    #辨識器捲積層名稱
    layer_name_list = ['my_con2d_1', 'my_prelu_1', 'my_bn_1', 'my_dropout_1',
                       'my_con2d_2', 'my_prelu_2', 'my_bn_2', 'my_dropout_2',
                       'my_con2d_3', 'my_prelu_3', 'my_bn_3', 'my_dropout_3']
    #建立產生器模型
    gmodel = generator_model()
    #建立辨識器模型
    dmodel = discriminator_model(window_size, layer_name_list)
    #建立分類器模型
    cmodel = classification_model(window_size, layer_name_list)
    #設定最佳化方式
    if first_training:
        gmodel_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
        dmodel_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)
    else:
        gmodel_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)
        dmodel_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.0001)
    #批次數量
    batch_size = 16
    image_count = 1
    if first_training:
        epochs = 15
    else:
        epochs = 200
    #影像二值化門檻
    gate_value = 0.15

    #匯入手寫數字圖片
    (train_image, train_label), (test_image, test_label) = tf.keras.datasets.mnist.load_data()
    #自定義資料產生器
    def image_generator(image, label, batch_size):
        #取得資料筆數
        max_index = len(label)
        index = 0
        #重複取值直到最後一筆資料
        while index < max_index:
            #決定這次bacth的最後一筆
            if index + batch_size >= max_index:
                end_index = (max_index - 1)
            else:
                end_index = index + batch_size

            #取得影像
            output_image = tf.convert_to_tensor(image[index:end_index, :, :])
            #將影像轉成0~1的float tensor
            output_image = tf.image.convert_image_dtype(output_image, tf.float32)
            #增加channel維度以符合模型輸入架構
            output_image = tf.expand_dims(output_image, -1)
            #取得影像標籤並轉成可以標示0~9的one hot編碼
            output_label = tf.one_hot(label[index:end_index], (label.max() + 1))
            #增加標籤tensor的維度以符合模型輸入格式
            dims_count = 0
            while dims_count < 2:
                output_label = tf.expand_dims(output_label, -1)
                dims_count += 1

            original_label = label[index:end_index]
            #更新取值起始位置
            index = end_index + 1

            yield output_image, output_label, original_label

    #權重複製
    def weight_copy(target_model, weight_model, layer_name_list):
        for layer_name in layer_name_list:
            tlayer = target_model.get_layer(name = layer_name)
            wlayer = weight_model.get_layer(name = layer_name)
            tlayer.set_weights(wlayer.get_weights())
            tlayer.trainable = False

    #GAN訓練主體
    def train_GAN(gmodel, gmodel_optimizer, dmodel, dmodel_optimizer,
                  train_image, train_label, batch_size, epochs, generator_train_data_path,
                  model_path, layer_name_list, first_training, gate_value):
        train_gmodel = False
        if first_training:
            cmodel = tf.keras.models.load_model(model_path + "c_model")
            weight_copy(dmodel, cmodel, layer_name_list)
        else:
            gmodel = tf.keras.models.load_model(model_path + "first_gmodel")
            dmodel = tf.keras.models.load_model(model_path + "first_dmodel")
            if train_gmodel:
                gmodel.trainable = True
                dmodel.trainable = False
            else:
                gmodel.trainable = False
                dmodel.trainable = True

        for epoch in range(epochs):
            g_loss_list = []
            d_loss_list = []
            print('Epoch:' + str(epoch))
        
            #從dataset batch匯入檔案
            for batch_image, batch_label, plot_label in image_generator(train_image, train_label, batch_size):
                g_loss, d_loss = model_training(gmodel, dmodel, gmodel_optimizer, dmodel_optimizer,
                                                batch_label, batch_image, first_training, train_gmodel, gate_value)
            
                g_loss_list.append(g_loss)
                d_loss_list.append(d_loss)
        
            
            print('G Model Loss:' + str(np.mean(g_loss_list)))
            print('D Model Loss:' + str(np.mean(d_loss_list)))
            if train_gmodel:
                #驗證g model目前能產生的結果
                generator_image(gmodel, batch_label, batch_image, generator_train_data_path, plot_label, epoch, gate_value)
                if np.mean(g_loss_list) < 0.06:
                    train_gmodel = False
                    gmodel.trainable = False
                    dmodel.trainable = True
            else:
                if np.mean(d_loss_list) < 0.001:
                    train_gmodel = True
                    gmodel.trainable = True
                    dmodel.trainable = False

        if first_training:
            gmodel.save(model_path + "first_gmodel")
            dmodel.save(model_path + "first_dmodel")
        else:
            gmodel.save(model_path + "second_gmodel")
            dmodel.save(model_path + "second_dmodel")

    #辨識器預先訓練
    def train_classification(cmodel, train_image, train_label, test_image, test_label, batch_size, epochs, model_path):
            cmodel.compile(optimizer = tf.keras.optimizers.Adam(),
                           loss = 'sparse_categorical_crossentropy', metrics = ['acc'])
            cmodel.fit(train_image, train_label, epochs = epochs, batch_size = batch_size,
                       validation_data=(test_image, test_label))
            cmodel.save(model_path + "c_model")
    
    Train_cmodel = False
    
    if Train_cmodel:
        train_classification(cmodel, train_image, train_label, test_image, test_label, batch_size, epochs, model_path)
    else:
        train_GAN(gmodel, gmodel_optimizer, dmodel, dmodel_optimizer,
                  train_image, train_label, batch_size, epochs, generator_train_data_path, model_path, layer_name_list,
                  first_training, gate_value)



        
            
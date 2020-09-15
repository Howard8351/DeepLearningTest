import tensorflow.keras.layers as layers
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras import Input
from tensorflow.keras import Model
import concurrent.futures
import time


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
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 700)])
            #確認虛擬GPU資訊    
            #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            #for logical_gpu in logical_gpus:
            #    tf.config.experimental.set_memory_growth(logical_gpu, True)
            #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except Exception as e:
    print(e)

def creat_clustering_model_pre(number_of_class, maxlen, input_dim, output_dim, layers_name):

    input_layer     = Input(shape = (maxlen,), name = layers_name[0])
    #input_length 是每筆輸入的長度
    #如果要接flatten的話是必須參數
    embedding_layer  = layers.Embedding(input_dim, output_dim, input_length = maxlen, name = layers_name[1])(input_layer)
    flatten_layer    = layers.Flatten(name = layers_name[2])(embedding_layer)
    hidden_layer     = layers.Dense(64, name = layers_name[3])(flatten_layer)
    activation_layer = layers.PReLU(name = layers_name[4])(hidden_layer)
    hidden_layer     = layers.Dense(128, name = layers_name[5])(activation_layer)
    activation_layer = layers.PReLU(name = layers_name[6])(hidden_layer)
    hidden_layer     = layers.Dense(256, name = layers_name[7])(activation_layer)
    activation_layer = layers.PReLU(name = layers_name[8])(hidden_layer)
    hidden_layer     = layers.Dense(512, name = layers_name[9])(activation_layer)
    activation_layer = layers.PReLU(name = layers_name[10])(hidden_layer)
    label_layer      = layers.Dense(number_of_class, activation = 'softmax', name = layers_name[11])(activation_layer)
    hidden_layer     = layers.Dense(512)(label_layer)
    activation_layer = layers.PReLU()(hidden_layer)
    hidden_layer     = layers.Dense(256)(activation_layer)
    activation_layer = layers.PReLU()(hidden_layer)
    hidden_layer     = layers.Dense(128)(activation_layer)
    activation_layer = layers.PReLU()(hidden_layer)
    hidden_layer     = layers.Dense(64)(activation_layer)
    activation_layer = layers.PReLU()(hidden_layer)
    output_layer     = layers.Dense(maxlen * output_dim)(activation_layer)
    #output_layer     = layers.PReLU()(hidden_layer)

    return Model(input_layer, [output_layer, flatten_layer, label_layer])

def creat_clustering_model(number_of_class, maxlen, input_dim, output_dim, layers_name):

    input_layer     = Input(shape = (maxlen,), name = layers_name[0])
    #input_length 是每筆輸入的長度
    #如果要接flatten的話是必須參數
    embedding_layer  = layers.Embedding(input_dim, output_dim, input_length = maxlen, name = layers_name[1])(input_layer)
    flatten_layer    = layers.Flatten(name = layers_name[2])(embedding_layer)
    hidden_layer     = layers.Dense(64, name = layers_name[3])(flatten_layer)
    activation_layer = layers.PReLU(name = layers_name[4])(hidden_layer)
    hidden_layer     = layers.Dense(128, name = layers_name[5])(activation_layer)
    activation_layer = layers.PReLU(name = layers_name[6])(hidden_layer)
    hidden_layer     = layers.Dense(256, name = layers_name[7])(activation_layer)
    activation_layer = layers.PReLU(name = layers_name[8])(hidden_layer)
    hidden_layer     = layers.Dense(512, name = layers_name[9])(activation_layer)
    activation_layer = layers.PReLU(name = layers_name[10])(hidden_layer)
    output_layer     = layers.Dense(number_of_class, activation = 'softmax', name = layers_name[11])(activation_layer)

    return Model(input_layer, [output_layer, flatten_layer])
    #return Model(input_layer, output_layer)
def creat_classification_model(number_of_class, maxlen, input_dim, output_dim, layers_name):
    input_layer     = Input(shape = (maxlen,), name = layers_name[0])
    #input_length 是每筆輸入的長度
    #如果要接flatten的話是必須參數
    embedding_layer  = layers.Embedding(input_dim, output_dim, input_length = maxlen, name = layers_name[1])(input_layer)
    flatten_layer    = layers.Flatten(name = layers_name[2])(embedding_layer)
    hidden_layer     = layers.Dense(8)(flatten_layer)
    activation_layer = layers.PReLU()(hidden_layer)
    dropout_layer    = layers.Dropout(0.5)(activation_layer)
    hidden_layer     = layers.Dense(16)(dropout_layer)
    activation_layer = layers.PReLU()(hidden_layer)
    dropout_layer    = layers.Dropout(0.5)(activation_layer)
    hidden_layer     = layers.Dense(4)(dropout_layer)
    activation_layer = layers.PReLU()(hidden_layer)
    output_layer     = layers.Dense(1, activation = 'sigmoid')(activation_layer)

    return Model(input_layer, output_layer)
#embedding預先訓練
def embedding_pre_training(classification_model, train_data, train_label, test_data, test_label, epochs, batch_size, model_path):
    classification_model.fit(x = train_data, y = train_label, batch_size = batch_size,
                             epochs = epochs, validation_data = (test_data, test_label))
    classification_model.save(model_path + "pre_training_embedding")

#模型預先訓練
def clustering_model_pre_training(clustering_model, train_data, batch_size, epochs, optimizer,
                                  loss_function, model_path, number_of_class, train_label):
    max_index = len(train_data)
    
    for epoch in range(epochs):
        start_index = 0
        index = 0
        loss_list = []
        crossentropy = []
        while index < max_index:
            if (start_index + batch_size) > max_index:
                index = max_index
            else:
                index = start_index + batch_size

            batch_data = train_data[start_index:index, :]
            batch_label = test_label[start_index:index]
            loss, predict_label, encode_data = premodel_fit(clustering_model, batch_data, optimizer, loss_function, number_of_class)
            
            loss_list.append(loss)
            start_index = index
            predict_label = tf.cast(tf.argmax(predict_label, axis = -1), dtype = tf.float32)
            batch_label = tf.convert_to_tensor(batch_label, dtype = tf.float32)
            crossentropy.append(tf.keras.losses.binary_crossentropy(batch_label, predict_label))
            

        mean_loss = np.mean(loss_list)
        print(f"Epoch: {epoch}  loss: {mean_loss}")
        crossentropy = np.mean(crossentropy)
        print(f"Epoch: {epoch}  crossentropy: {crossentropy}")
        if epoch > 1 and epoch % 100 == 0:
            clustering_model.save(model_path + "pre_training_model")

    clustering_model.save(model_path + "pre_training_model")

#模型訓練
def clustering_model_training(clustering_model, train_data, batch_size, epochs, optimizer, loss_function,
                              model_path, number_of_class, output_epoch, random_seed, train_label):
    max_index = len(train_data)
    class_centor = []
    np.random.seed(random_seed)

    for index in range(number_of_class):
        #class_centor.append(train_data[index, :])
        class_centor.append(train_data[np.random.randint(max_index), :])

    class_centor = tf.convert_to_tensor(class_centor)
    centor_predict_label, class_centor = clustering_model(class_centor)
    
    for epoch in range(epochs):
        start_index = 0
        index = 0
        loss_list = []
        predict_label_list = []
        encode_data_list   = []
        while index < max_index:
            if (start_index + batch_size) > max_index:
                index = max_index
            else:
                index = start_index + batch_size

            batch_data = train_data[start_index:index, :]
            loss, predict_label, encode_data = model_fit(clustering_model, batch_data, optimizer, loss_function, number_of_class, class_centor)
            if start_index == 0:
                predict_label_list = predict_label
                encode_data_list   = encode_data
            else:
                predict_label_list = tf.concat([predict_label_list, predict_label], axis = 0)
                encode_data_list   = tf.concat([encode_data_list, encode_data], axis = 0)
            #predict_label = tf.argmax(predict_label, axis = -1)

            loss_list.append(loss)

            start_index = index

        mean_loss = np.mean(loss_list)
        print(f"Epoch: {epoch}  loss: {mean_loss}")
        a = tf.reduce_sum(tf.abs(tf.math.subtract(train_label, predict_label_list)))
        class_centor = centor_update(predict_label_list, encode_data_list, class_centor, number_of_class)

    clustering_model.save(model_path + "model_" + str(output_epoch))

def centor_update(predict_label, encode_data, class_centor, number_of_class):
    predict_label = tf.one_hot(predict_label, depth = number_of_class)
    predict_label = tf.split(predict_label, num_or_size_splits = number_of_class, axis = -1)
    new_class_centor = []
    for index in range(number_of_class):
        count = tf.reduce_sum(predict_label[index])
        if count > 0:
            new_centor = tf.reduce_sum(predict_label[index] * encode_data, axis = 0) / count
            new_class_centor.append(new_centor)
        else:
            new_class_centor.append(tf.split(class_centor, num_or_size_splits = number_of_class, axis = 0)[index])
    class_centor = tf.convert_to_tensor(new_class_centor)
    return class_centor

@tf.function
def premodel_fit(model, batch_data, optimizer, loss_function, number_of_class):
    with tf.GradientTape() as tape:
        decode_data, encode_data, predict_label = model(batch_data)
        loss = tf.reduce_sum(tf.abs(tf.math.subtract(encode_data, decode_data)))

    weight = model.trainable_weights
    gradients = tape.gradient(loss, weight)
    optimizer.apply_gradients(zip(gradients, weight))
    
    return loss, predict_label, encode_data

@tf.function
def model_fit(model, batch_data, optimizer, loss_function, number_of_class, class_centor):
    with tf.GradientTape() as tape:
        predict_label, encode_data = model(batch_data)
        #predict_label = tf.split(predict_label, num_or_size_splits = len(predict_label), axis = -1)
        #predict_label = tf.cast(tf.argmax(predict_label, axis = -1), dtype = tf.float32)

        class_centor_list = tf.split(class_centor, num_or_size_splits = number_of_class, axis = 0)
        cluster_similarity_matrix = []
        cluster_label = []
        for index in range(number_of_class):
            #for data_index in range(len(batch_data)):
            cluster_similarity = tf.math.abs(encode_data - class_centor_list[index])
            cluster_label.append(tf.reduce_mean(cluster_similarity, axis = -1))
            cluster_similarity_matrix.append(tf.reduce_mean(cluster_similarity, axis = -1))

        cluster_label = tf.transpose(tf.convert_to_tensor(cluster_label, dtype = tf.float32))
        cluster_label = tf.argmin(cluster_label, axis = -1)
        cluster_similarity_matrix = tf.convert_to_tensor(cluster_similarity_matrix, dtype = tf.float32)
        cluster_similarity_matrix = cluster_similarity_matrix * tf.transpose(predict_label)
        
        loss = tf.reduce_mean(cluster_similarity_matrix)

    weight = model.trainable_weights
    gradients = tape.gradient(loss, weight)
    optimizer.apply_gradients(zip(gradients, weight))
    
    return loss, cluster_label, encode_data


def clustering_test(clustering_model, test_data, test_label, batch_size, loss_function):
    max_index = len(test_data)
    start_index = 0
    index = 0
    mae = []
    
    while index < max_index:
        if (start_index + batch_size) > max_index:
            index = max_index
        else:
            index = start_index + batch_size

            batch_data  = test_data[start_index:index, :]
            batch_label = test_label[start_index:index]

            #predict_label, encode_data = clustering_model.predict(batch_data)
            predict_label = clustering_model_predict(clustering_model, batch_data)
            predict_label = tf.cast(tf.argmax(predict_label, axis = -1), dtype = tf.float32)
            batch_label = tf.convert_to_tensor(batch_label, dtype = tf.float32)
            #batch_label = tf.reshape(tf.convert_to_tensor(batch_label, dtype = tf.float32), shape = [-1, 1])
            #predict = tf.reshape(predict, shape = [-1, 1])
            mae.append(loss_function(batch_label, predict_label))
        start_index = index

    mae = np.mean(mae)
    print(f"Mae: {mae}")

@tf.function
def clustering_model_predict(clustering_model, batch_data):
    predict_label, encode_data = clustering_model(batch_data)
    return predict_label

def classification_test(clustering_model, test_data, test_label, batch_size, loss_function):
    max_index = len(test_data)
    start_index = 0
    index = 0
    mae = []
    
    while index < max_index:
        if (start_index + batch_size) > max_index:
            index = max_index
        else:
            index = start_index + batch_size

            batch_data  = test_data[start_index:index, :]
            batch_label = test_label[start_index:index]

            #predict_label = clustering_model.predict(batch_data)
            predict_label = classification_model_predict(clustering_model, batch_data)
            predict_label = tf.squeeze(tf.round(predict_label))
            batch_label = tf.convert_to_tensor(batch_label, dtype = tf.float32)
            #batch_label = tf.reshape(tf.convert_to_tensor(batch_label, dtype = tf.float32), shape = [-1, 1])
            #predict = tf.reshape(predict, shape = [-1, 1])
            mae.append(tf.keras.losses.binary_crossentropy(batch_label, predict_label))
        start_index = index

    mae = np.mean(mae)
    print(f"Mae: {mae}")

@tf.function
def classification_model_predict(clustering_model, batch_data):
    predict_label = clustering_model(batch_data)
    return predict_label

#權重複製
def weight_copy(target_model, weight_model, layer_name_list):
    index = 0

    if len(layer_name_list) > 1:
        for layer_name in layer_name_list:
            tlayer = target_model.get_layer(name = layer_name)
            wlayer = weight_model.get_layer(name = layer_name)
            tlayer.set_weights(wlayer.get_weights())
            if index < 2:
                tlayer.trainable = False
                index += 1
    else:
        layer_name = layer_name_list[0]
        tlayer = target_model.get_layer(name = layer_name)
        wlayer = weight_model.get_layer(name = layer_name)
        tlayer.set_weights(wlayer.get_weights())
        tlayer.trainable = False

def model_training(number_of_class, maxlen, input_dim, output_dim, layers_name,
                   train_data, batch_size, epochs, loss_function, model_path, epoch, train_label, random_seed):
    pre_training_model = tf.keras.models.load_model(model_path + "pre_training_model")
    clustering_model   = creat_clustering_model(number_of_class, maxlen, input_dim, output_dim, layers_name)
    weight_copy(clustering_model, pre_training_model, layers_name)
    clustering_test(clustering_model, train_data, train_label, batch_size, loss_function)
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001)
    clustering_model_training(clustering_model, train_data, batch_size, epochs, optimizer, loss_function, model_path,
                              number_of_class, epoch, random_seed, train_label)
    return str(epoch)+' Complete'


#if model_pre_training:
#    clustering_model   = creat_clustering_model_pre(number_of_class, maxlen, input_dim, output_dim, layers_name)
#else:
#    pre_training_model = tf.keras.models.load_model(model_path + "pre_training_model")
#    clustering_model   = creat_clustering_model(number_of_class, maxlen, input_dim, output_dim, layers_name)
#    #weight_copy(clustering_model, pre_training_model, layers_name[0:3])
#    weight_copy(clustering_model, pre_training_model, layers_name)
#if model_pre_training:
#    clustering_model_pre_training(clustering_model, train_data, batch_size, epochs, optimizer, loss_function, model_path, number_of_class)
#else:
#    clustering_model_training(clustering_model, train_data, batch_size, epochs, optimizer, loss_function, model_path, number_of_class)
#    clustering_test(clustering_model, test_data, test_label, batch_size, loss_function)

if __name__ == '__main__':
    
    #匯入資料
    max_features = 10000
    maxlen = 20
    #num_word 代表只保留訓練資料集文章中較常出現的前N個單字
    #在imdb 中每個單字都會用一個整數編號代表
    #每篇文章用一個list儲存
    (train_data, train_label), (test_data, test_label) = imdb.load_data(num_words = max_features)
    #將資料轉換成固定長度的2Dtensor
    train_data = preprocessing.sequence.pad_sequences(train_data, maxlen = maxlen, truncating = 'post')
    test_data = preprocessing.sequence.pad_sequences(test_data, maxlen = maxlen, truncating = 'post')

    #embedding 可以學到的詞彙量
    input_dim = max_features
    #每個詞彙會用幾個特徵表示
    output_dim = 20
    layers_name = ["input_layer", "embedding_layer", "flatten_layer", "Dense_layer_1", "Prelu_layer_1",
                   "Dense_layer_2", "Prelu_layer_2", "Dense_layer_3", "Prelu_layer_3",
                   "Dense_layer_4", "Prelu_layer_4", "Dense_layer_5"]
    model_pre_training = False
    optimizer       = tf.keras.optimizers.Adam(learning_rate = 0.00001)
    loss_function   = tf.keras.losses.mean_absolute_error
    batch_size      = 8
    epochs          = 200
    model_path      = "E:/python program/Deep_learning_clustering/"
    number_of_class = 2

    if model_pre_training:
        batch_size = 32
        epochs     = 140
        classification_model = creat_classification_model(number_of_class, maxlen, input_dim, output_dim, layers_name)
        classification_model.compile(loss = tf.keras.losses.binary_crossentropy,
                                     optimizer = tf.keras.optimizers.Adam(learning_rate = 0.00001), metrics = ['accuracy'])
        embedding_pre_training(classification_model, train_data, train_label, test_data, test_label, epochs, batch_size, model_path)
    else:
        #pre_training_embedding = tf.keras.models.load_model(model_path + "pre_training_embedding")
        ##classification_test(pre_training_embedding, train_data, train_label, batch_size, loss_function)
        #clustering_model = creat_clustering_model_pre(number_of_class, maxlen, input_dim, output_dim, layers_name)
        #weight_copy(clustering_model, pre_training_embedding, layers_name[0:2])
        #clustering_model_pre_training(clustering_model, train_data, batch_size, epochs, optimizer, loss_function, model_path, number_of_class, train_label)
    
        #多執行緒數量
        number_of_process = 5
        np.random.seed(round(time.time()))
        random_seed_list = np.random.randint(100000, size = number_of_process)
        #註冊多執行緒管理器
        executor = concurrent.futures.ProcessPoolExecutor(max_workers = number_of_process)
        #保留宣告的執行緒以便取得結果
        process_list = []
        number_of_model = 5
       
        for epoch in range(number_of_model):
            model_training(number_of_class, maxlen, input_dim, output_dim, layers_name,
                       train_data, batch_size, epochs, loss_function, model_path, epoch, train_label, random_seed_list[epoch])
            process_list.append(executor.submit(model_training, number_of_class, maxlen, input_dim, output_dim, layers_name,
                                                train_data, batch_size, epochs, loss_function, model_path, epoch, train_label,
                                                random_seed_list[epoch]))
        for process in concurrent.futures.as_completed(process_list):
            print(process)        
        input('Enter')

        #number_of_model = 5
        #for epoch in range(number_of_model):
        #    clustering_model = tf.keras.models.load_model(model_path + "model_" + str(epoch))
        #    print("model_" + str(epoch))
        #    clustering_test(clustering_model, train_data, train_label, batch_size, loss_function)
        #    clustering_test(clustering_model, test_data, test_label, batch_size, loss_function)


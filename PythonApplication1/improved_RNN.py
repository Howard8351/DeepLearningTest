import os
#匯入檔案

data_dir = 'D:\python program\jena_climate_2009_2016'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
f = open(fname)
data = f.read()
f.close()
#將匯入的字串以分行符號分割
lines = data.split('\n')
#第一行是資料的標頭，取出後以，分割
header = lines[0].split(',')
lines = lines[1:]
print(header)
print(len(lines))

#做資料前處理
import numpy as np
#第一個參數是紀錄時間，因為不用來參考所以排除
float_data = np.zeros((len(lines), len(header) - 1))
#將資料以列為單位取出
for i, line in enumerate(lines):
    #將取出的字串分割後轉成浮點數保留，並去掉記錄日期
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values

#要跑多處理程序的話畫圖會出錯
#待修改
#if __name__ == '__main__':
#    #畫出溫度的部分
#    from matplotlib import pyplot as plt
#    temp = float_data[:, 1]
#    #全畫
#    plt.plot(range(len(temp)), temp)
#    plt.title('All Temp')
#    #plt.legend()
#    plt.figure()
#    #畫前1440筆(每10分鐘紀錄一次，約為10天的資料)
#    plt.plot(range(1440), temp[:1440])
#    plt.title('10 Days Temp')
#    #plt.show()
#    plt.close()

#對訓練資料做正規畫，並保留正規畫用的參數
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

#因為資料是每10分鐘取樣一次，因此相鄰的資料差異不大
#所以建立一個資料產生器來幫我們產生取樣頻率為小時的資料
def generator(data, lookback, delay, min_index, max_index,
              shuffle=False, batch_size=128, step=6):
    #如果沒有定義max_index則設為資料長度減預測目標的延遲量
    if max_index is None:
        max_index = len(data) - delay - 1
    #這段是為了確保即便i取道最小值時也能有足夠的舊資料能參考
    i = min_index + lookback
    while 1:
        #根據shuffle決定資料要隨機抽樣還是按照順序產生
        if shuffle:
            rows = np.random.randint(
                min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        #rows的長度是目前預期每次訓練匯入的資料量
        #每一筆資料是包含現在到過去10天的資料，而預設是每十分鐘取樣一次，目前採用每小時重新取樣一次
        #所以每一筆資料有lookback // step
        #最後則是每次取樣包含的特徵數量
        #list的index如果用負數代表倒數過來第幾筆
        samples = np.zeros((len(rows),
                            lookback // step,
                            data.shape[-1]))
        targets = np.zeros((len(rows),))
        #將訓練資料和預測值根據取樣的index取出
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        #在function中如果包含yield就會被定義成一個產生器
        #而yield就像是一個不會終止function的return
        yield samples, targets

##每筆資料可以參考的過去資料量
#lookback = 1440
##每幾筆資料取樣一次
#step = 6
##要預測多久以後的溫度
#delay = 144
##每次訓練要匯入幾筆資料
#batch_size = 128
##訓練資料產生器(採用隨取樣)
#train_gen = generator(float_data,
#                      lookback=lookback,
#                      delay=delay,
#                      min_index=0,
#                      max_index=200000,
#                      shuffle=True,
#                      step=step,
#                      batch_size=batch_size)
##驗證資料產生器(依序從min_index到max_index產生資料)
#val_gen = generator(float_data,
#                    lookback=lookback,
#                    delay=delay,
#                    min_index=200001,
#                    max_index=300000,
#                    step=step,
#                    batch_size=batch_size)
##測試資料產生器(依序從min_index到max_index產生資料)
#test_gen = generator(float_data,
#                     lookback=lookback,
#                     delay=delay,
#                     min_index=300001,
#                     max_index=None,
#                     step=step,
#                     batch_size=batch_size)
##計算驗證資料產生器執行多少次才能跑完驗證資料集
#val_steps = (300000 - 200001 - lookback)
##計算測試資料產生器執行多少次才能跑完測試資料集
#test_steps = (len(float_data) - 300001 - lookback)

#在建立一個模型之前經常會先用一個簡易的方式先試著解決問題
#而深度學型模型的效能要超越該方式才能證明是較有效的
#下面是一個簡單的溫度預測方式
#假設接下來24小時內的溫度都會和現在一樣
#並用MAE來驗證誤差程度
#改寫成平行化增加效率

def evaluate_naive_method(float_data, lookback, delay,
                         min_index, max_index, step, batch_size):
    batch_maes = []
    val_gen = generator(float_data,
                    lookback = lookback,
                    delay = delay,
                    min_index = min_index,
                    max_index = max_index,
                    step = step,
                    batch_size = batch_size)
    val_steps = (max_index - min_index - lookback)
    for step in range(val_steps):
        #print('執行狀況: ',step, '/', val_steps, sep = '')
        samples, targets = next(val_gen)
        preds = samples[:, -1, 1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    return batch_maes

import concurrent.futures
numberofthread = 8

#用下列條件式避免跑多處理程序時執行不必要的部分
if __name__ == '__main__':
    #用多處理程序加速基本概念驗證
    #with concurrent.futures.ProcessPoolExecutor(max_workers = numberofthread) as executor:
    #    batch_mases =[]
    #    min_index=200001
    #    max_index=300000
    #    val_thread_index = []

    #    for index in range(min_index, (max_index + 1), int((max_index - min_index) / numberofthread)):
    #        val_thread_index.append(index)
    #    val_thread_index[-1] = max_index
        
    #    processlist = []
    #    for i in range(len(val_thread_index)-1):
    #        if(i == 1):
    #            processlist.append(executor.submit(evaluate_naive_method, float_data, lookback,
    #                                           delay, val_thread_index[i], val_thread_index[i + 1], step, batch_size))
    #        else:
    #            processlist.append(executor.submit(evaluate_naive_method, float_data, lookback,
    #                                           delay, (val_thread_index[i] + 1), val_thread_index[i + 1], step, batch_size))
    #    for s in concurrent.futures.as_completed(processlist):
    #        batch_mases = batch_mases + s.result()
    #import time
    #wait = True
    #while wait:
    #    try:
    #        print('簡單概念MAE:', np.mean(batch_mases) * std[1])
    #        wait = False
    #    except:
    #        print('計算中')
    #        time.sleep(5)
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import RMSprop
    
    ##嘗試用密集連接網路來預測溫度
    #model = sequential([
    #    layers.flatten(input_shape=(lookback // step, float_data.shape[-1])),
    #    layers.dense(32, activation='relu'),
    #    layers.dense(1)])
    #model.compile(optimizer=rmsprop(), loss='mae')
    #history = model.fit_generator(train_gen,
    #                              steps_per_epoch=500,
    #                              epochs=20,
    #                              validation_data=val_gen,
    #                              validation_steps=val_steps)
    
    ##嘗試用回歸網路預測溫度
    #model = Sequential([
    #    layers.GRU(32, input_shape=(None, float_data.shape[-1])),
    #    layers.Dense(1)])
    #model.compile(optimizer=RMSprop(), loss='mae')
    #history = model.fit_generator(train_gen,
    #                              steps_per_epoch=500,
    #                              epochs=20,
    #                              validation_data=val_gen,
    #                              validation_steps=val_steps)
    
    ##在回歸網路中加入dropout以避免過度訓練
    #model = Sequential([
    #        layers.GRU(32, dropout = 0.2, recurrent_dropout=0.2,
    #                  input_shape=(None, float_data.shape[-1])),
    #        layers.Dense(1)])
    #model.compile(optimizer=RMSprop(), loss='mae')
    #history = model.fit_generator(train_gen,
    #                              steps_per_epoch=500,
    #                              epochs=40,
    #                              validation_data=val_gen,
    #                              validation_steps=val_steps)
    ##增加回歸網路層數來改善誤差
    #model = Sequential([
    #        layers.GRU(32, dropout = 0.1, recurrent_dropout=0.5,
    #                   return_sequences = True,
    #                  input_shape=(None, float_data.shape[-1])),
    #        layers.GRU(64, dropout = 0.1, recurrent_dropout=0.5),
    #        layers.Dense(1)])
    #model.compile(optimizer=RMSprop(), loss='mae')
    #history = model.fit_generator(train_gen,
    #                              steps_per_epoch=500,
    #                              epochs=40,
    #                              validation_data=val_gen,
    #                              validation_steps=val_steps)

    #在溫度預測中把資料翻轉後再訓練的效果並不理想
    #因為對溫度預測來說越新的資料越重要，所以新的資料要最後學習比較合理
    #而對於語言來說，單字出現在文章前面還是後面沒有絕對的意義
    #接下來試試把imdb的評論資料反轉後再學習
    from tensorflow.keras.datasets import imdb
    from tensorflow.keras.preprocessing import sequence
    #max_features = 10000
    #maxlen = 500
    #(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    #x_train = [x[::-1] for x in x_train]
    #x_test = [x[::-1] for x in x_test]
    #x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    #x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    #model = Sequential([layers.Embedding(max_features, 128),
    #                   layers.LSTM(32),
    #                   layers.Dense(1, activation='sigmoid')])
    #model.compile(optimizer='rmsprop',
    #              loss='binary_crossentropy',
    #              metrics=['acc'])
    #history = model.fit(x_train, y_train,
    #                    epochs=10,
    #                    batch_size=128,
    #                    validation_split=0.2)

    ##試著用雙向回歸網路來做imdb評論分類
    ##tensorflow用Bidirectional來建立雙向回歸網路
    ##第一個參數必須是一個回歸網路架構
    ##如果沒有指定反向學習的回歸網路架構，Bidirectional會建立一個跟順向學習回歸網路相同的架構
    ##也能指定兩個網路的結果合併的方式
    #model = Sequential([layers.Embedding(max_features, 32),
    #                   layers.Bidirectional(layers.LSTM(32)),
    #                   layers.Dense(1, activation='sigmoid')])
    #model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
    #history = model.fit(x_train, y_train,
    #                    epochs=10,
    #                    batch_size=128,
    #                    validation_split=0.2)
    ##用雙向回歸網路預測溫度
    #model = Sequential([
    #    layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])),
    #    layers.Dense(1),
    #    ])
    #model.compile(optimizer=RMSprop(), loss='mae')
    #history = model.fit_generator(train_gen,
    #                              steps_per_epoch=500,
    #                              epochs=40,
    #                              validation_data=val_gen,
    #                              validation_steps=val_steps)

    #用卷積網路處理序列資料
    #匯入資料並把資料切成同樣大小的tensor
    max_features = 10000
    max_len = 500
    print('Loading data...')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Pad sequences (samples x time)')
    x_train = sequence.pad_sequences(x_train, maxlen=max_len)
    x_test = sequence.pad_sequences(x_test, maxlen=max_len)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras import layers
    from tensorflow.keras.optimizers import RMSprop
    ##建立1D卷積網路來學習序列資料
    #model = Sequential([
    #    layers.Embedding(max_features, 128, input_length=max_len),
    #    layers.Conv1D(32, 7, activation='relu'),
    #    layers.MaxPooling1D(5),
    #    layers.Conv1D(32, 7, activation='relu'),
    #    layers.GlobalMaxPooling1D(),
    #    layers.Dense(1)])

    #model.summary()
    #model.compile(optimizer=RMSprop(lr=1e-4),
    #              loss='binary_crossentropy',
    #              metrics=['acc'])
    #history = model.fit(x_train, y_train,
    #                    epochs=10,
    #                    batch_size=128,
    #                    validation_split=0.2)

    #model = Sequential()
    #model.add(layers.Conv1D(32, 5, activation='relu',
    #                        input_shape=(None, float_data.shape[-1])))
    #model.add(layers.MaxPooling1D(3))
    #model.add(layers.Conv1D(32, 5, activation='relu'))
    #model.add(layers.MaxPooling1D(3))
    #model.add(layers.Conv1D(32, 5, activation='relu'))
    #model.add(layers.GlobalMaxPooling1D())
    #model.add(layers.Dense(1))

    #model.compile(optimizer=RMSprop(), loss='mae')
    
    #val_steps = 500
    #try:
    #    history = model.fit_generator(train_gen,
    #                              steps_per_epoch=500,
    #                              epochs=20,
    #                              validation_data=val_gen,
    #                              validation_steps=val_steps)
    #except Exception as e:
    #    print(e)

    #提高資料的取樣率
    #每筆資料可以參考的過去資料量
    lookback = 1440
    #每幾筆資料取樣一次
    step = 3
    #要預測多久以後的溫度
    delay = 144
    #每次訓練要匯入幾筆資料
    batch_size = 128
    #訓練資料產生器(採用隨取樣)
    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=200000,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)
    #for i in range(500):
    #    samples, targets = next(train_gen)
    #    if(np.sum(np.isnan(samples)) > 0):
    #        print('資料異常')
    #    if(np.sum(np.isnan(targets)) > 0):
    #        print('目標異常')

    #驗證資料產生器(依序從min_index到max_index產生資料)
    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=200001,
                        max_index=300000,
                        step=step,
                        batch_size=batch_size)

    #for i in range(500):
    #    samples, targets = next(val_gen)
    #    if(np.sum(np.isnan(samples)) > 0):
    #        print('資料異常')
    #    if(np.sum(np.isnan(targets)) > 0):
    #        print('目標異常')
    #測試資料產生器(依序從min_index到max_index產生資料)
    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=300001,
                         max_index=None,
                         step=step,
                         batch_size=batch_size)
    #計算驗證資料產生器執行多少次才能跑完驗證資料集
    val_steps = (300000 - 200001 - lookback)
    #計算測試資料產生器執行多少次才能跑完測試資料集
    test_steps = (len(float_data) - 300001 - lookback)

    #val_steps = 500
    ##嘗試用回歸網路預測溫度
    #model = Sequential([
    #    layers.GRU(32, dropout = 0.1, recurrent_dropout=0.2,
    #               input_shape=(None, float_data.shape[-1])),
    #    layers.Dense(1)])
    #model.compile(optimizer=RMSprop(), loss='mae')
    #history = model.fit_generator(train_gen,
    #                              steps_per_epoch=500,
    #                              epochs=40,
    #                              validation_data=val_gen,
    #                              validation_steps=val_steps)

    ##加入卷積網路做rnn的前處理
    #model = Sequential([
    #    layers.Conv1D(32, 5, activation='relu',
    #                  input_shape=(None, float_data.shape[-1])),
    #    layers.MaxPooling1D(3),
    #    layers.Conv1D(32, 5, activation='relu'),
    #    layers.GRU(32, recurrent_dropout=0.05),
    #    layers.Dense(1)])

    #model.summary()
    #model.compile(optimizer=RMSprop(), loss='mae')


    #val_steps = 500
    #history = model.fit_generator(train_gen,
    #                              steps_per_epoch=500,
    #                              epochs=20,
    #                              validation_data=val_gen,
    #                              validation_steps=val_steps)

    #用雙向回歸網路預測溫度
    model = Sequential([
        layers.Bidirectional(layers.GRU(32), input_shape=(None, float_data.shape[-1])),
        layers.Dense(1),
        ])
    model.compile(optimizer=RMSprop(), loss='mae')
    val_steps = 500
    history = model.fit_generator(train_gen,
                                  steps_per_epoch=500,
                                  epochs=20,
                                  validation_data=val_gen,
                                  validation_steps=val_steps)


    

    
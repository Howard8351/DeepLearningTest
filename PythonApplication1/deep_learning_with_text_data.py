#透過keras內建的function對文字做oen-hot編碼(可以針對單字等級或字元等級)
from tensorflow.keras.preprocessing.text import Tokenizer
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
#建立一個新的標籤器(num_words代表只考慮前幾個最常出現的單字，此外還有其他初始化參數)
tokenizer = Tokenizer(num_words=1000)
#根據給予的list建立單字的index表
tokenizer.fit_on_texts(samples)
#根據先前建立的index表把輸入list中的單字轉成index後輸出
sequences = tokenizer.texts_to_sequences(samples)
#根據先前建立的index表把輸入list轉成矩陣後輸出(根據指定的模式決定轉成甚麼類型的矩陣)
#這裡轉成只有0,1的二元矩陣
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')
#輸出目前建立的單字index表
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

#oen-hot編碼加入hash函數以避免index表過大導致產生的稀疏矩陣過大
import numpy as np
samples = ['The cat sat on the mat.', 'The dog ate my homework.']
dimensionality = 1000
max_length = 10
results = np.zeros((len(samples), max_length, dimensionality))
for i, sample in enumerate(samples):
    for j, word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality
        results[i, j, index] = 1.

#用keras建立world Embedding
#相對於one-hot編碼world Embedding可以用較小的向量表示更多的資料
#world Embedding是透過從資料中學習來產生的，而不是透過固定的編法方式產生
#world Embedding可以藉由自己的資料訓練，或是載入別人預先訓練好的
from tensorflow.keras.layers import Embedding
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing

#第一個參數決定最多能學幾種單字
#第二個參數決定輸出維度
embedding_layer = Embedding(1000, 64)

#匯入資料
max_features = 10000
maxlen = 20
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words = max_features)
print(x_train.shape)
print(type(x_train))
print(x_train[0][:7])
#將資料轉換成固定長度的2Dtensor
x_train = preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen, truncating = 'post')
x_test = preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen, truncating = 'post')
print(x_train[0][:7])

from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
model = Sequential([
    #Embedding之後要用flatten的話需要設定input_length
    #但是只有在每個輸入的長度都固定時才適用
    Embedding(10000, 8, input_length=maxlen),
    layers.Flatten(),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop', 
              loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_split= 0.2)

#接下來要使用別人預先訓練的Embedding
#匯入imdb的原始檔案重新作標籤化
import os
imdb_dir = 'D:/python program/aclImdb/'
train_dir = os.path.join(imdb_dir, 'train/')
labels = []
texts = []
#匯入原始文字檔並加上評價標籤
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(train_dir, label_type)
    dirlist_count = 1
    list_length = len(os.listdir(dir_name))
    for fname in os.listdir(dir_name):
        #確認fname從倒數第4個元素開始到最後是否是.txt
        if fname[-4:] == '.txt':
            #print(fname)
            f = open(os.path.join(dir_name, fname), encoding = 'utf8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
        print('Label Type:', label_type, end = '\t')
        print('Number of file:', dirlist_count, '/', list_length)
        dirlist_count += 1

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
#每個評論只保留100個單字
maxlen = 100
training_samples = 200
validation_samples = 10000
#最多只學習10000種單字
max_words = 10000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))
#只保留前100個單字
data = pad_sequences(sequences, maxlen=maxlen, truncating = 'post')
#將評論標籤由list轉成ndarray
labels = np.asarray(labels)
print('Shape of data tensor:', data.shape)
print('Shape of label tensor:', labels.shape)
#產生一個大小跟評論數量相同的ndarray(預設從0到給定數值-1)
indices = np.arange(data.shape[0])
#將indices隨機排列
np.random.shuffle(indices)
#根據新的index從新排序資料並分成訓練和驗證資料
data = data[indices]
labels = labels[indices]
x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples: training_samples + validation_samples]
y_val = labels[training_samples: training_samples + validation_samples]

#匯入並確認預先訓練的embeddings架構
glove_dir = 'D:/python program/glove.6B/'
embeddings_index = {}
f = open(os.path.join(glove_dir, 'glove.6B.100d.txt'), encoding = 'utf8')
#開啟文字檔後一行一行匯入
for line in f:
    #將字串分割成list
    values = line.split()
    #每一行的第一個字是後面權重代表的單字
    word = values[0]
    #將權重list轉成ndarray
    coefs = np.asarray(values[1:], dtype='float32')
    #建立單字的權重字典
    embeddings_index[word] = coefs
f.close()
print('Found %s word vectors.' % len(embeddings_index))

#建立embedding權重矩陣
#預先訓練的架構是用100個數值表示一個單字
embedding_dim = 100
embedding_matrix = np.zeros((max_words, embedding_dim))
#從我們建立的單字索引表一個個尋找目前的單字是否被訓練過
for word, i in word_index.items():
    #確保沒有異常值
    if i < max_words:
        #根據單字尋找權重
        embedding_vector = embeddings_index.get(word)
        #如果有找到在相對應的index保留權重
        #如果找不到該列會都是零
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Flatten, Dense
#建立自己的分類模型
model = Sequential([
    Embedding(max_words, embedding_dim, input_length=maxlen),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()
#把Embedding層的權重換成預先訓練的
#並設成不可訓練，避免被變更
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False
#定義最佳化函數
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
#匯入訓練和驗證資料訓練模型
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))
#匯出模型
model.save_weights('pre_trained_glove_model.h5')

#建立新的模型來比較沒有匯入預先訓練的權重會差多少
model = Sequential([
    Embedding(max_words, embedding_dim, input_length=maxlen),
    Flatten(),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['acc'])
history = model.fit(x_train, y_train,
                    epochs=10,
                    batch_size=32,
                    validation_data=(x_val, y_val))

#匯入測試資料集
test_dir = os.path.join(imdb_dir, 'test/')
labels = []
texts = []
for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    dirlist_count = 1
    list_length = len(os.listdir(dir_name))
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname), encoding = 'utf8')
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)
        print('Label Type:', label_type, end = '\t')
        print('Number of file:', dirlist_count, '/', list_length)
        dirlist_count += 1

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)
#匯入剛才保存的模型驗證預測結果
model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)
import tensorflow.keras as keras
import numpy as np
#取得要訓練的文章
path = keras.utils.get_file('nietzsche.txt',
                            origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
text = open(path, encoding='UTF-8').read().lower()
print('Corpus length:', len(text))

#定義每個句子做多有幾個字元
maxlen = 60
#每隔幾個字元重新取樣新的句子
step = 3
#訓練資料
sentences = []
#目標資料
next_chars = []
#挑選訓練和測試資料集
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('Number of sequences:', len(sentences))
#將原始資料集中出現過的字元都只保留唯一一個並排序
chars = sorted(list(set(text)))
print('Unique characters:', len(chars))
#建立字元字典已進行one hot編碼
char_indices = dict((char, chars.index(char)) for char in chars)
print('Vectorization...')
#隊訓練和測試資料集做one hot編碼
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import Input

#建立模型和最佳化
input_sentence = Input(shape = (maxlen, len(chars)))
layer = layers.LSTM(128)(input_sentence)
output_layer = layers.Dense(len(chars), activation='softmax')(layer)
model = Model(input_sentence, output_layer)

optimizer = keras.optimizers.RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)

#從分類結果挑選下一個字元
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

import random
import sys,gc

#利用迴圈來產生句子並訓練模型
for epoch in range(1, 60):
    print('epoch', epoch)
    model.fit(x, y, batch_size=128, epochs=1)
    start_index = random.randint(0, len(text) - maxlen - 1)
    original_generated_text = text[start_index: start_index + maxlen]
    print('--- Generating with seed: "' + original_generated_text + '"')
    for temperature in [0.2, 0.5, 1.0, 1.2]:
        print('------ temperature:', temperature)
        sys.stdout.write(original_generated_text)
        generated_text = original_generated_text
        for i in range(400):
            sampled = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(generated_text):
                sampled[0, t, char_indices[char]] = 1.
            preds = model.predict(sampled, verbose=0)[0]
            #用來舒緩model.predict目前有記憶體溢位的問題
            keras.backend.clear_session()
            next_index = sample(preds, temperature)
            next_char = chars[next_index]
            generated_text += next_char
            generated_text = generated_text[1:]
            sys.stdout.write(next_char)
        print('')


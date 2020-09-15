from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers
from tensorflow.keras import Input
#用過去常用的序列方式建立模型
seq_model = Sequential()
seq_model.add(layers.Dense(32, activation='relu', input_shape=(64,)))
seq_model.add(layers.Dense(32, activation='relu'))
seq_model.add(layers.Dense(10, activation='softmax'))
seq_model.summary()

#試著用functional的方式建立一樣的模型
#用functional的方式建立模型時會把相對的輸入(前面建好的各層模型)，放在新建的層後面然後存在一個變數中
#Model是functional api建立模型的方式
#第一個參數決定輸入資料格式
#而第二個參數會根據前面一層層給予的定義去建立模型
#但是輸入資料的格式要和在堆疊模型時定義的一樣，不然會回傳錯誤
input_tensor = Input(shape=(64,))
x = layers.Dense(32, activation='relu')(input_tensor)
x = layers.Dense(32, activation='relu')(x)
output_tensor = layers.Dense(10, activation='softmax')(x)
model = Model(input_tensor, output_tensor)
model.summary()
#模型建立好後定義最佳化方式和訓練的方法和之前一樣
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
import numpy as np
#建立隨機亂數來測試模型是否能運作
x_train = np.random.random((1000, 64))
y_train = np.random.random((1000, 10))
model.fit(x_train, y_train, epochs=10, batch_size=128)
score = model.evaluate(x_train, y_train)

#試著建立一個有多個輸入的模型
text_vocabulary_size = 10000
question_vocabulary_size = 10000
answer_vocabulary_size = 500
#建立第一個輸入層
#因為不能確定每個輸入文章的長度，所以輸入大小是None
text_input = Input(shape=(None,), dtype='int32', name='text')
embedded_text = layers.Embedding(text_vocabulary_size, 64)(text_input)
encoded_text = layers.LSTM(32)(embedded_text)
#建立第二個輸入層
question_input = Input(shape=(None,), dtype='int32', name='question')
embedded_question = layers.Embedding(question_vocabulary_size, 32)(question_input)
encoded_question = layers.LSTM(16)(embedded_question)
#將剛才建立的兩個輸入層以最後一層為基準合併
concatenated = layers.concatenate([encoded_text, encoded_question], axis=-1)
answer = layers.Dense(answer_vocabulary_size, activation='softmax')(concatenated)
model = Model([text_input, question_input], answer)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy', metrics=['acc'])

#建立隨機文字編碼作為訓練資料
num_samples = 1000
max_length = 100
text = np.random.randint(1, text_vocabulary_size, size=(num_samples, max_length))
question = np.random.randint(1, question_vocabulary_size, size=(num_samples, max_length))
#答案是一個One-hot編碼，不是整數序列
answers = np.random.randint(0, 1, size=(num_samples, answer_vocabulary_size))
#有兩種方式來訓練多輸入的模型
#第一種:將輸入依序建成一個lsit
#則模型會根據剛才輸入層中lsit的順序對應到現在lsit中的元素
model.fit([text, question], answers, epochs=10, batch_size=128)
#第二種:建立一個字典，而輸入資料的名稱要對應到相對的輸入層名稱
#這個方法只有在輸入層有定義名稱時才能用
model.fit({'text': text, 'question': question}, answers, epochs=10, batch_size=128)

#試著建立多輸出模型
vocabulary_size = 50000
num_income_groups = 10
posts_input = Input(shape=(None,), dtype='int32', name='posts')
embedded_posts = layers.Embedding(256, vocabulary_size)(posts_input)
x = layers.Conv1D(128, 5, activation='relu')(embedded_posts)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.Conv1D(256, 5, activation='relu')(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation='relu')(x)
#第一個輸出
age_prediction = layers.Dense(1, name='age')(x)
#第二個輸出
income_prediction = layers.Dense(num_income_groups
                                 ,activation='softmax',name='income')(x)
#第三個輸出
gender_prediction = layers.Dense(1, activation='sigmoid', name='gender')(x)
#一樣用lsit建立多輸出模型
model = Model(posts_input,[age_prediction, income_prediction, gender_prediction])
#而在多輸出模型中需要對不同的輸出節點各自定義最佳化和計算loss的方式
#一樣可以用lsit或字典來定義個個輸出的loss和最佳化方式
model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'])
model.compile(optimizer='rmsprop',
              loss={'age': 'mse',
                    'income': 'categorical_crossentropy','gender': 'binary_crossentropy'})
#要注意在多輸出模型中不同的損失函數會因為本身算出來的數值較大(例如用於預測問題的MSE通常會落在3~5
#，而用於分類問題的crossentropy可以到0.1或更低)而被模型是為需要優先調整的項目
#這可能會導致其他的輸出在訓練過程中被忽略而無法得到較好的結果
#可以透過設定各個損失函數的權重來平衡各損失函數對模型的重要性
#一樣可以用lsit或字典來定義個個輸出的損失函數權重
model.compile(optimizer='rmsprop',
              loss=['mse', 'categorical_crossentropy', 'binary_crossentropy'],
              loss_weights=[0.25, 1., 10.])
model.compile(optimizer='rmsprop',
              loss={'age': 'mse','income': 'categorical_crossentropy','gender': 'binary_crossentropy'},
              loss_weights={'age': 0.25, 'income': 1., 'gender': 10.})
#一樣可以用lsit或字典來定義個個輸出訓練目標
#model.fit(posts, [age_targets, income_targets, gender_targets],
#          epochs=10, batch_size=64)
#model.fit(posts,
#          {'age': age_targets, 'income': income_targets, 'gender': gender_targets},
#          epochs=10, batch_size=64)

#用functional API實現Directed acyclic graphs of layers
#這種架構的特點是一個層的輸出不能做為自己的輸入
#只有循環層(RNN)能在內部做循環
#這裡假設輸入是一個4Dtensor
x = Input(shape=(150, 150, 3), dtype='int32', name='image_imput')
branch_a = layers.Conv2D(128, 1, activation='relu', strides=2)(x)
branch_b = layers.Conv2D(128, 1, activation='relu')(x)
branch_b = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_b)
branch_c = layers.AveragePooling2D(3, strides=2)(x)
branch_c = layers.Conv2D(128, 3, activation='relu')(branch_c)
branch_d = layers.Conv2D(128, 1, activation='relu')(x)
branch_d = layers.Conv2D(128, 3, activation='relu')(branch_d)
branch_d = layers.Conv2D(128, 3, activation='relu', strides=2)(branch_d)
output = layers.concatenate([branch_a, branch_b, branch_c, branch_d], axis=-1)

#用functional API實現Residual connections
#這種作法經常用在層數較多的模型
#可以用來避免前幾層學到的特徵在後面的網路中失去代表性
#作法是將前面學到的權重和後面的權重相加
#如果權重的形狀已經不同可以透過線性轉換成相同大小(如沒有啟動函數的dense或沒有啟動函數的1*1卷積網路)
#這裡假設輸入是一個4Dtensor
#x = ...
#y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
#y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
#y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
#新舊權重大小相同直接相加
#y = layers.add([y, x])

#x = ...
#y = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
#y = layers.Conv2D(128, 3, activation='relu', padding='same')(y)
#y = layers.MaxPooling2D(2, strides=2)(y)
#新舊權重大小不同，做線性轉換改變形狀
#residual = layers.Conv2D(128, 1, strides=2, padding='same')(x)
#y = layers.add([y, residual])

#用functional API實現 layer sharing
#如果要處理的多個輸入是有高度相關性的(例如判斷兩個句子的意思是否相近)
#就不用建立兩模型分別學型相似的資料
#可以只用一個模型對兩個輸入資料做訓練

#先建立共用的層
lstm = layers.LSTM(32)
#分別接上兩個輸入
left_input = Input(shape=(None, 128))
left_output = lstm(left_input)
right_input = Input(shape=(None, 128))
right_output = lstm(right_input)
#將輸入的架構合併
merged = layers.concatenate([left_output, right_output], axis=-1)
#接上輸出層
predictions = layers.Dense(1, activation='sigmoid')(merged)
#model = Model([left_input, right_input], predictions)
#model.fit([left_data, right_data], targets)

#在tensorflow中可以向使用layer一樣把預先建立的模型加入其他的網路中
#例如要處理一個雙攝影機的輸入資料
#可以把兩個輸入接在同一個卷積網路上

from tensorflow.keras import applications

#xception是tensorflow中的一個影像處理模型
xception_base = applications.Xception(weights=None, include_top=False)

left_input = Input(shape=(250, 250, 3))
right_input = Input(shape=(250, 250, 3))
left_features = xception_base(left_input)
right_input = xception_base(right_input)
merged_features = layers.concatenate([left_features, right_input], axis=-1)
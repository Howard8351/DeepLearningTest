import tensorflow as tf
import tensorflow.data as tf_dataset
import time
from tensorflow.keras import layers
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras import Sequential

#estimator 是tensorflow 底下的高階模型表示方式
#提供簡單的擴展和異步訓練能力
#已經可以處理大多數的常規問題

#實作預設的estimator

#第一步建立輸入函式
#function 必須有兩個輸出 feature 和 label
#feature 是一個字典每個key代表一種特徵
#label 是樣本的類別或目標值
#建議使用tf.data.dataset來完成
#但是只要能符合輸出格式就行

feature_column = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth']
label_column = 'Species'
labels = ['Setosa', 'Versicolor', 'Virginica']
train_data = 'E:/python program/estimator 範例/iris_training.csv'
test_data = 'E:/python program/estimator 範例/iris_test.csv'
batch_size = 32

#建立資料輸入函式
def input_fun(data_path, label_column, training = True, batch_size = 32):
    csv_dataset = tf.data.experimental.make_csv_dataset(
        file_pattern = data_path,
        #設定每次dataset要取幾筆資料較適合搭配take來取資料
        #但是之後要用batch 來取資料的話建議設為1 
        batch_size = 1,
        #標記Label 的column name
        label_name = label_column,
        #而外標記要被當成 Na的 string
        na_value = "?",
        #表示匯入的資料是否要隨機排序
        shuffle = False,
        #表示資料要重複匯入幾次
        #如果為None就循環輸入
        num_epochs=1,
        #表示是否忽略文件中有錯誤的資料
        ignore_errors=True)
    if training:
        csv_dataset = csv_dataset.shuffle(1000).repeat()

    return csv_dataset.batch(batch_size)

#建立feature_column list
#etsimater 需要提供feature_cloumn list 來表示哪些column 是輸入資料
#tf.feature_column 有提供多種類別的資料形式
#要自己根據輸入資料類別做選擇
feature_column_list = []
for feature in feature_column:
    feature_column_list.append(tf.feature_column.numeric_column(key = feature))


#建立estimator
#tensorflow 提供多種預設的 estimatosr
#這裡選用 DNNClassifier
classifier = tf.estimator.DNNClassifier(
    #給予定義的feature_cloumn
    feature_columns = feature_column_list,
    #定義模型架構和節點數
    hidden_units = [5, 3],
    #定義資料集的類別數量
    n_classes = 3,
    optimizer = 'Adam')

#使用estimator 進行訓練
#input_fn 參數被要求用 lambda 的方式來包裹輸入函式
#steps 表示要訓練幾次
start_time = time.time()
classifier.train(input_fn = lambda: input_fun(train_data, label_column, training = True),
                 steps = 5000)
end_time = time.time()
estimator_cost_time = end_time - start_time

#評估訓練結果
eval_result = classifier.evaluate(
    input_fn=lambda: input_fun(test_data, label_column, training = False))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

#建立一個keras model 來比較運算時間
feature_column_list = []
for feature in feature_column:
    feature_column_list.append(tf.feature_column.numeric_column(key = feature))

#tf.data.experimental.make_csv_dataset 的輸出分為兩部分
#第一個為以各個輸入column name為 key的字典
#第二個是包含數筆輸入資料類別的 eager_tensor 
#需要用DenseDeature 來取的資料
#
model = Sequential([layers.DenseFeatures(feature_column_list),
                   layers.Dense(5, 'relu'),
                   layers.Dense(3, 'relu'),
                   layers.Dense(3, 'softmax')])

input_layer_dict = {}
for feature in feature_column:
    input_layer_dict[feature] = Input(shape = (1,), name = feature)

input_layer = layers.DenseFeatures(feature_column_list)(input_layer_dict)
layer = layers.Dense(5, 'relu')(input_layer)
layer = layers.Dense(3, 'relu')(layer)
output_layer = layers.Dense(3, 'softmax')(layer)

model = Model(input_layer_dict, output_layer)

model.compile(
    loss = tf.keras.losses.sparse_categorical_crossentropy,
    optimizer = 'adam',
    metrics = ['accuracy'])

model.summary()

def fit_input(data_path, label_column, batch_size = 32):
    csv_dataset = tf.data.experimental.make_csv_dataset(
        file_pattern = data_path,
        #設定每次dataset要取幾筆資料較適合搭配take來取資料
        #但是之後要用batch 來取資料的話建議設為1 
        batch_size = batch_size,
        #標記Label 的column name
        label_name = label_column,
        #而外標記要被當成 Na的 string
        na_value = "?",
        #表示匯入的資料是否要隨機排序
        shuffle = False,
        #表示資料要重複匯入幾次
        #如果為None就循環輸入
        num_epochs=1,
        #表示是否忽略文件中有錯誤的資料
        ignore_errors=True)
    
    return csv_dataset.shuffle(1000)

a = fit_input(train_data, label_column)

for features, labels in a.take(4):
    numlayer = layers.DenseFeatures(feature_column_list)
    print(numlayer(features).numpy())
    print()

#  print(features)
#  print()
#  print(labels)

start_time = time.time()
model.fit(fit_input(train_data, label_column), epochs = 1250)
end_time = time.time()
print("Keras Model cost {} sec".format(end_time - start_time))
print("Estimator cost {} sec".format(estimator_cost_time))


import os, shutil

#原始資料路徑
original_dataset_dir = 'E:/python program/dogs-vs-cats/'
#挑選出來的資料存放路徑
base_dir = 'E:/python program/dogs-vs-cats/dogs-vs-cats-resample/'

train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
test_cats_dir = os.path.join(test_dir, 'cats')
test_dogs_dir = os.path.join(test_dir, 'dogs')

try:
    #如果資料夾不存在則建立資料夾
    os.mkdir(base_dir)
    os.mkdir(train_dir)
    os.mkdir(validation_dir)
    os.mkdir(test_dir)
    os.mkdir(train_cats_dir)
    os.mkdir(train_dogs_dir)
    os.mkdir(validation_cats_dir)
    os.mkdir(validation_dogs_dir)
    os.mkdir(test_cats_dir)
    os.mkdir(test_dogs_dir)
except FileExistsError:
    print('資料夾已存在')

##將圖片複製到指定路徑以便訓練和驗證
#fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
#original_traindataset_dir = os.path.join(original_dataset_dir, 'train/')
#for fname in fnames:
#    src = os.path.join(original_traindataset_dir, fname)
#    dst = os.path.join(train_cats_dir, fname)
#    shutil.copyfile(src, dst)
#fnames = ['cat.{}.jpg'.format(i) for i in range(1000, 1500)]
#for fname in fnames:
#    src = os.path.join(original_traindataset_dir, fname)
#    dst = os.path.join(validation_cats_dir, fname)
#    shutil.copyfile(src, dst)
#fnames = ['cat.{}.jpg'.format(i) for i in range(1500, 2000)]
#for fname in fnames:
#    src = os.path.join(original_traindataset_dir, fname)
#    dst = os.path.join(test_cats_dir, fname)
#    shutil.copyfile(src, dst)
    
#fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
#for fname in fnames:
#    src = os.path.join(original_traindataset_dir, fname)
#    dst = os.path.join(train_dogs_dir, fname)
#    shutil.copyfile(src, dst)
#fnames = ['dog.{}.jpg'.format(i) for i in range(1000, 1500)]
#for fname in fnames:
#    src = os.path.join(original_traindataset_dir, fname)
#    dst = os.path.join(validation_dogs_dir, fname)
#    shutil.copyfile(src, dst)
#fnames = ['dog.{}.jpg'.format(i) for i in range(1500, 2000)]
#for fname in fnames:
#    src = os.path.join(original_traindataset_dir, fname)
#    dst = os.path.join(test_dogs_dir, fname)
#    shutil.copyfile(src, dst)
    
import tensorflow as tf
from tensorflow.keras import layers

#建立模型
model = tf.keras.Sequential([
    layers.Conv2D(32, (2,2), activation= 'relu', input_shape=(150,150,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
  ])


model.summary()

from tensorflow.keras import optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#設定模型最佳化方式
model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

#建立一個影像產生器
#可以用來即時產生影像給模型訓練
#在這裡用來匯入影像給模型訓練和驗證
#並且正規化成0~1
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

#設定影像產生器的影像來源路徑、
#輸出影像大小、每次匯入數量和影像資料的類別型態
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

#確認匯入資料格式
for data_batch, labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

#fit_generator會一直匯入資料作訓練直到跑完steps_per_epoch指定的次數才會跑下一次epoch
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)
#將訓練好的模型存成獨立檔案
model.save('cats_and_dogs_small_1.h5')

#將訓練和驗證結果視覺化
import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
#plt.show()

#建立新的影像資料產生器
#下列參數會對匯入的影像作隨機的型變
#以增加資料的變化性來避免過度訓練
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

import tensorflow.keras.preprocessing.image as image

fnames = [os.path.join(train_cats_dir, fname) for
fname in os.listdir(train_cats_dir)]
img_path = fnames[3]
img = image.load_img(img_path, target_size=(150, 150))
x = image.img_to_array(img)
x = x.reshape((1,) + x.shape)
i = 0
for batch in datagen.flow(x, batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i % 4 == 0:
        break
plt.show()


#建立新的模型並新增Dropout層，以便進一步改善過度訓練
#Dropout 會根據設定的機率將部分的輸入資料設為0
model = tf.keras.Sequential([
    layers.Conv2D(32, (3,3), activation= 'relu', input_shape=(150,150,3)),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(64, (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Conv2D(128, (3,3), activation= 'relu'),
    layers.MaxPooling2D((2,2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(512, activation = 'relu'),
    layers.Dense(1, activation = 'sigmoid')
  ])
model.compile(loss='binary_crossentropy',
    optimizer=optimizers.RMSprop(lr=1e-4),
    metrics=['acc'])

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,)

#注意測試資料不能做型變
#以便符合未來可能的輸入格式
test_datagen = ImageDataGenerator(rescale=1./255)
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')
validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary')
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50)

model.save('cats_and_dogs_small_2.h5')

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc) + 1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
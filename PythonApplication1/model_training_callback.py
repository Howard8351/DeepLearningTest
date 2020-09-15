import tensorflow.keras as keras

#在一個list中可以定義多個callback
callbacks_list = [
    keras.callbacks.EarlyStopping(
        #監看的對象
        monitor='acc',
        #連續幾次epoch監看目標沒有改善就不再訓練
        patience=1,),
    keras.callbacks.ModelCheckpoint(
        #給定模型儲存的位置和名稱
        filepath='my_model.h5',
        #監看的對象
        monitor='val_loss',
        #設為True時只要監看對象沒有改善就不覆寫模型
        save_best_only=True,)]

#這裡要注意的是callback要監看的對象都必須被定義在模型最佳化和訓練方式中
#model.compile(optimizer='rmsprop',
#              loss='binary_crossentropy',
#              metrics=['acc'])

#model.fit(x, y,
#          epochs=10,
#          batch_size=32,
#          callbacks=callbacks_list,
#          validation_data=(x_val, y_val))

callbacks_list = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        #當callback觸發時學習率減少比例
        factor=0.1,
        #決定監看目標在幾次epoch後都沒有改善就啟動callback(不確定是否要連續)
        patience=10,)]

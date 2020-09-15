import gym
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np



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
                gpu, [tf.config.experimental.VirtualDeviceConfiguration(memory_limit = 800)])
        #確認虛擬GPU資訊    
        #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        #for logical_gpu in logical_gpus:
        #    tf.config.experimental.set_memory_growth(logical_gpu, True)
        #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
except Exception as e:
        print(e)

#建立遊戲環境
env = gym.make('CartPole-v0')

print(tf.keras.backend.floatx())
tf.keras.backend.set_floatx('float64')
print(tf.keras.backend.floatx())
            
def creat_model():
    #匯入模型
    return load_model('E:/python program/增強式學習結果/Model/DQL_Model')

def play_loop(epochs):
    DQL_model = creat_model()
    #取得環境狀態的數量
    num_states = env.observation_space.shape[0]
    win_count = 0
   
    for epoch in range(epochs):
        #讓 environment 重回初始狀態
        state = env.reset()
        #統計在一次epoch中總共做了計次動作才結束
        step_times = 0

        while True:
            #呈現 environment
            #env.render()
            #取得模型選擇的動作
            action_rate = DQL_model.predict(np.reshape(state, [1, num_states]))
            action = np.argmax(action_rate)
            #在這次的環境中根據給予的動作會得到四個回傳值
            # next_state:互動後的新環境
            # reward:新環境給予的回饋值
            # done:是否已達到環境的結束條件
            #action = train_model.get_action(state, random_action_rate)
            next_state, reward, done, info = env.step(action)
            
            step_times += 1
            #如果完成遊戲 win_count +1
            if step_times == 200:
                win_count += 1
        
            #如果 done 結束迴圈
            if done:
                print('Epoch:{} Steps:{}'.format(epoch, step_times))
                break
            #更新環境資訊
            state = next_state
    
    print('遊戲勝率:{}'.format(win_count / epochs))



play_loop(200)
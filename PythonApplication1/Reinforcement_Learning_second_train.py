import gym
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import concurrent.futures
import math


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

env = gym.make('CartPole-v0')
a = state = env.reset()

#取得動作種類的數量
num_actions = env.action_space.n
#取得環境狀態的數量
num_states = env.observation_space.shape[0]

print(tf.keras.backend.floatx())
tf.keras.backend.set_floatx('float64')
print(tf.keras.backend.floatx())

class Deep_Q_Learning:
    def __init__(self, num_states, num_actions, gamma,
                 memory_capacity, min_memory, batch_size, learning_rate):
        #每個state包含的參數數量
        self.num_states = num_states
        #可以選擇的action數量
        self.num_actions = num_actions
        #計算next_q_value的期望值
        #數值越大對next_q_value越重視
        self.gamma = gamma
        #保留舊資料做訓練
        #每一筆memory資料包含 state, action, reward, next_state
        self.memory = {'state': [], 'action': [], 'reward': [], 'next_state': [], 'done': []}
        #最多要保留幾筆紀錄
        self.memory_capacity = memory_capacity
        #最少保留幾筆紀錄後開始做訓練
        self.min_memory = min_memory
        #每次訓練的取樣數量
        self.batch_size = batch_size
        #目前主流的Deep Q learning會用兩個一樣的模型來做訓練
        #只對train_model做訓練
        #target_model只被動接收權重
        #訓練模型
        self.train_model = creat_model()
        #目標模型
        self.target_model = creat_model()
        #設定最佳化方式
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)
        #設定損失函數
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.loss_value = None

    def get_action(self, state, random_action_rate):
        if np.random.random() < random_action_rate:
            action = np.random.randint(0, self.num_actions)
        else:
            #action_rate = self.train_model(np.reshape(state, [1, self.num_states]))
            action_rate = self.train_model.predict(np.reshape(state, [1, self.num_states]))
            action = np.argmax(action_rate)

        return action

    def save_memory(self, new_memory):
        #如果紀錄滿了丟掉最舊的
        if len(self.memory['state']) >= self.memory_capacity:
            for key in self.memory.keys():
                self.memory[key].pop(0)
        #新增紀錄
        for key, value in new_memory.items():
            self.memory[key].append(value)

    def get_memory_size(self):
        return len(self.memory['state'])

    def get_loss_value(self):
        return self.loss_value

    #對於在eager model中的tensorflow運算可以在function前面加上@tf.function來改善運算效率
    #但是該function會不能用debug監看
    @tf.function
    def calculate_gradient(self, train_state, train_action, nextq_value):
        #在GradientTape中計算loss_value以便計算梯度
        with tf.GradientTape() as tape:
            train_model_output = self.train_model(train_state)
            q_value = tf.math.reduce_sum(train_model_output
                                         * tf.one_hot(train_action, self.num_actions, dtype = 'float64'), axis=1)
            loss_value = self.loss_function(nextq_value, q_value)
        #計算梯度
        weight = self.train_model.trainable_variables
        gradients = tape.gradient(loss_value, weight)
        #根據梯度更新權重
        self.optimizer.apply_gradients(zip(gradients, weight))
        #self.loss_value = loss_value.numpy()


    def training_model(self):
        if len(self.memory['state']) > self.min_memory:
            #取得一次批量的訓練資料
            sample_index = np.random.choice(len(self.memory['state']), self.batch_size)
            train_state =  np.asarray([self.memory['state'][index] for index in sample_index])
            train_action = np.asarray([self.memory['action'][index] for index in sample_index])
            train_reward = np.asarray([self.memory['reward'][index] for index in sample_index])
            train_next_state = np.asarray([self.memory['next_state'][index] for index in sample_index])
            train_done = np.asarray([self.memory['done'][index] for index in sample_index])
            #取的目標模型對next_state的預測結果
            #taeget_predict = np.max(self.target_model(train_next_state), axis = 1)
            taeget_predict = np.max(self.target_model.predict(train_next_state), axis = 1)
            #計算next_q value
            #如果選擇的動作會導致done發生就直接輸出reward，不考慮next_state帶來的回饋
            #nextq_value = train_reward + (self.gamma * taeget_predict)
            nextq_value = np.where(train_done, train_reward, train_reward + (self.gamma * taeget_predict))
            
            self.calculate_gradient(train_state, train_action, nextq_value)
            
            
    def copy_weight(self):
        #將Train Model的權重複製到Target Model
        self.target_model.set_weights(self.train_model.get_weights())

    def save_model(self):
        self.train_model.save('E:/python program/增強式學習結果/Model/DQL_Model_second_train',
                              include_optimizer = False)
            
def creat_model():
    #匯入模型
    return load_model('E:/python program/增強式學習結果/Model/DQL_Model')

def training_loop(epochs, num_states, num_actions, gamma, random_action_rate, target_replace_count, memory_capacity,
                min_memory, batch_size, learning_rate):
    DQL_model = Deep_Q_Learning(num_states, num_actions, gamma,
                                memory_capacity, min_memory, batch_size, learning_rate)
    
    step_list = []
    reward_list = []
    step_mean_list = []
    loss_list = []
    target_step = 0

    #建立一個loop先把最少memory需求補齊
    #讓 environment 重回初始狀態
    state = env.reset()
    #統計在一次epoch中總共做了計次動作才結束
    step_times = 0

    while DQL_model.get_memory_size() < (min_memory - 1):
        #取得模型選擇的動作
        action = DQL_model.get_action(state, random_action_rate)
        #在這次的環境中根據給予的動作會得到四個回傳值
        # next_state:互動後的新環境
        # reward:新環境給予的回饋值
        # done:是否已達到環境的結束條件
        #action = train_model.get_action(state, random_action_rate)
        next_state, reward, done, info = env.step(action)

        #theta單位是弧度
        #和theta_threshold_radians 相同
        x, v, theta, omega = next_state

        ##改善reward所代表的意義以提升訓練效果
        ##小車離中間越近越好
        #r1 = ((env.x_threshold - abs(x)) / env.x_threshold) * 0.2
        ##柱子越正越好
        #r2 = ((env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians) * 0.8
        #reward = r1 + r2
        step_times =+ 1

        if done:
            reward = 0
        if step_times == 200:
            reward = 1
            DQL_model.save_model()
            #建立環境經驗
            new_memory = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': False}
        else:
            #建立環境經驗
            new_memory = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}

        DQL_model.save_memory(new_memory)
        #更新環境資訊
        if done:
            step_times = 0
            state = env.reset()
        else:
            state = next_state

    #print(len(DQL_model.get_memory()['state']))
    for epoch in range(epochs):
        #讓 environment 重回初始狀態
        state = env.reset()
        #累計各epoch中的reward
        rewards = 0 
        #統計在一次epoch中總共做了計次動作才結束
        step_times = 0
        
        loss = []

        while True:
            #呈現 environment
            #env.render()
            #取得模型選擇的動作
            action = DQL_model.get_action(state, random_action_rate)
            #在這次的環境中根據給予的動作會得到四個回傳值
            # next_state:互動後的新環境
            # reward:新環境給予的回饋值
            # done:是否已達到環境的結束條件
            #action = train_model.get_action(state, random_action_rate)
            next_state, reward, done, info = env.step(action)

            #theta單位是弧度
            #和theta_threshold_radians 相同
            x, v, theta, omega = next_state

            ##改善reward所代表的意義以提升訓練效果
            ##小車離中間越近越好
            #r1 = ((env.x_threshold - abs(x)) / env.x_threshold) * 0.2
            ##柱子越正越好
            #r2 = ((env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians) * 0.8
            #reward = r1 + r2
            
            step_times += 1
            target_step += 1
            
            if done:
                reward = 0
            if step_times == 200:
                reward = 1
                DQL_model.save_model()
                #建立環境經驗
                new_memory = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': False}
            else:
                #建立環境經驗
                new_memory = {'state': state, 'action': action, 'reward': reward, 'next_state': next_state, 'done': done}

            #計算這次epoch中的reward總和
            rewards += reward


            #將現有資訊封裝後放入記憶體以便之後訓練
            DQL_model.save_memory(new_memory)
            #有足夠的資料後開始訓練
            DQL_model.training_model()
            #loss_value = DQL_model.get_loss_value()
            #if loss_value != None:
            #    loss.append(loss_value)
            
            #如果step_times超過門檻就複製模型權重
            #if target_step >= target_replace_threshold:
            if (target_step % target_replace_count) == 0:
                DQL_model.copy_weight()
                target_step = 0
        
            if done:
                #loss_list.append(np.mean(loss))
                step_list.append(step_times)
                reward_list.append(rewards)
                print('Episode: {} Steps: {} Rewards: {}'.format(epoch, step_times, rewards))
                #print('Loss: {}'.format(np.mean(loss)))
                if (epoch % 50 == 0) and (epoch > 49):
                    #if np.mean(step_list[-100:]) > 100:
                    #    target_replace_threshold = int(np.mean(step_list[-100:]) * increase_rate)
                    step_mean_list.append(np.mean(step_list[-50:]))
                #    print('Last 100 Episode: {} Mean Step: {} timesteps, Mean_rewards {}'.format(epoch, np.mean(step_list[-100:])
                #                                                                                 , np.mean(reward_list[-100:])))
                #    #print('Mean loss: {}'.format(np.mean(loss_list[-100:])))
                break
            #更新環境資訊
            state = next_state
    
    return step_list, step_mean_list, learning_rate



if __name__ == '__main__':
    learning_rate = 0.001
    #每次取樣的數量
    batch_size = 32
    #隨機選擇動作的機率
    #讓模型能有機會學到不同動作帶來的反饋
    random_action_rate = 0.1
    #對next q_value的期望值
    gamma = 0.9
    #target network 更新間隔
    target_replace_count = 50
    #Q learning的訓練會保留一定數量的結果在對目標模型做訓練
    #這樣能讓模型更容易完成收斂
    #決定要保留多少次的輸入和輸出結果
    memory_capacity = 2500
    min_memory = 50
    #進行遊戲的次數
    epochs = 200

    step_list, hidden, learning_rate = training_loop(epochs, num_states, num_actions, gamma, random_action_rate,
                                                     target_replace_count, memory_capacity, min_memory, batch_size, learning_rate)

    ##處理程序數量
    #number_of_process = 5
    ##註冊多process管理器
    #executor = concurrent.futures.ProcessPoolExecutor(max_workers = number_of_process)
    #if True:
    ##for hidden_index in range(len(hidden_list)):
    #    #註冊的process列表
    #    processlist = []

    #    for i in range(5):
    #        processlist.append(executor.submit(training_loop, epochs, num_states, num_actions, hidden, gamma, random_action_rate,
    #                                                 reduce_rate, min_rate, target_replace_count, target_replace_threshold,increase_rate,
    #                                                 max_replace_threshold, memory_capacity, min_memory, batch_size, learning_rate))
    #    plot_count = 0
    #    for process in concurrent.futures.as_completed(processlist):
    #        step_list, step_mean_list, learning_rate = process.result()  
    #    #for i in range(5):
    #    #    step_list, step_mean_list, learning_rate = training_loop(epochs, num_states, num_actions, hidden, gamma, random_action_rate,
    #    #                                                             reduce_rate, min_rate, target_replace_count, target_replace_threshold,increase_rate,
    #    #                                                             max_replace_threshold, memory_capacity, min_memory, batch_size, learning_rate)
    #        plt.figure(figsize=(16,9))
    #        plt.subplot(2,1,1)
            
    #        title = 'Model_with_[7, 5]_hidden_layer  ' + str(plot_count)
    #        plt.title(title)
    #        plt.xlabel('Epoch')
    #        plt.ylabel('Steps in each Epoch')
    #        plt.plot(range(len(step_list)), step_list,
    #                 label='Step')
    #        plt.legend()

    #        plt.subplot(2,1,2)
    #        title = 'Model_with_[7, 5]_hidden_layer  ' + str(plot_count)
    #        plt.title(title)
    #        plt.xlabel('Epoch')
    #        plt.ylabel('Mean Steps per 50 Epoch')
    #        plt.plot(range(len(step_mean_list)), step_mean_list,
    #                 label='Mean Step')
    #        plt.legend()

    #        plt.savefig('E:/python program/增強式學習結果/增強式學習結果' + title + '.png')
    #        plt.close()
    #        plot_count += 1
   
    

    #input('pause Enter')
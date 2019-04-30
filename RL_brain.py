import numpy as np
import pandas as pd
from maze_env import Maze       # 环境模块，暂不深究


class QLearningTable:
    def __init__(self, actions, learning_rate=0.01, reward_decay=0.9, e_greedy=0.9):
        self.actions = actions      # 供选择的动作列表，为[0, 1, 2, 3]，代表上下左右
        self.lr = learning_rate     # 学习效率
        self.gamma = reward_decay   # 奖励衰减值
        self.epsilon = e_greedy     # 贪婪度
        self.q_table = pd.DataFrame(columns=self.actions, dtype=np.float64)     # 创建Q表，共四列，列名为0、1、2、3

    def choose_action(self, observation):
        self.check_state_exist(observation)     # 检测本state是否存在于q_table中，不在则插入

        # 贪婪度=0.9，90%的可能按照Q表的最优值选择行为
        if np.random.uniform() < self.epsilon:
            state_action = self.q_table.loc[observation, :]     # []中先行后列，以逗号分割；冒号表所有行/列
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)     # 先选最大值的行为，若多个行为有同一个最大值，再随机选一个
        # 贪婪度=0.9，10%的可能随机选行为
        else:
            action = np.random.choice(self.actions)

        return action   # 0-3的4个数，分别对应上下左右4个动作

    def learn(self, s, a, r, s_):
        '''
        Q表中的Q值为Q估计值；
        Q现实值要通过公式计算而来，Q现实 = (Qmax * gamma) + R；
        Q值的更新也要通过公式计算，新值 = |Q估计-Q现实| * alpha；
        （Q现实值只在更新Q值时用到，若不加强调，Q值默认指Q估计值）
        详情：http://21guns.top/2019/04/13/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0-Q-learning/

        '''
        self.check_state_exist(s_)      # 检查新位置是否在Q表中，选择动作之前检查的是当前位置（即旧位置）
        q_predict = self.q_table.loc[s, a]  # 根据state和action的交叉，找到状态对应动作的Q值，此是预测值（与估计值同义）
        
        # 如果新位置没有标记结束，即不是地狱或天堂，则根据公式来计算Q现实值
        if s_ != 'terminal': 
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # 目标值/现实值 = 奖励 + （奖励衰减值 * 本state对应4个action中最大的Q值）
        # 若新位置标记结束，即进入地狱/天堂，此时Q现实直接等于奖励r，因为新位置是结束标记，其state不会出现在Q表中，自然没法找到Qmax，自然不用公式计算
        else:
            q_target = r

        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # 更新对应的state-action值，即Q值；新值 = |估计-现实| * 学习率

    def check_state_exist(self, state):
        '''
        共有4x4=16个格子，2地狱1天堂，三者可视为同种，即结束state
        经过多次训练后，Q表最多可有14条纪录，16-3+1=14
        '''
        if state not in self.q_table.index:         # 若本状态（即当前所在格子的位置）未在Q表中，则插入相关记录，
            self.q_table = self.q_table.append(     # 增加一个Series；Series单一列/行；DataFrame中包含一个或多个Series；每个Series均有一个名称
                pd.Series(                          # 本series表一行：在当前格子位置，上下左右分别对应的Q值（列名为动作，行名为状态，交叉即Q值）
                    [0]*len(self.actions),          # 内容值，可通过index访问，默认为0，即默认的Q值为0；
                    index=self.q_table.columns,     # 将列作为索引项，为[0, 1, 2, 3]，可理解为键值对中的键
                    name=state,                     # Series的名称
                )
            )

if __name__ == "__main__":
    '''
    测试所用，并无其他意义
    '''
    env = Maze()
    RL = QLearningTable(actions=list(range(env.n_actions)))
    observation = env.reset()   # [5.0, 5.0, 35.0, 35.0]

    print("Q表：\n", RL.q_table)

    action = RL.choose_action(str(observation))
    observation_, reward, done = env.step(action)

    print("动作：\n", action)
    print("观测值：\n", observation)
    print("新观测值：\n", observation_)
    print("奖励：\n", reward)
    print("done：\n", done)
    print("新Q表：\n", RL.q_table)





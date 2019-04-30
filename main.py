from maze_env import Maze       # 环境模块，只看思路，暂不深究代码实现
from RL_brain import QLearningTable     # RL的大脑，负责决策和思考

def update():
    for episode in range(100):      # 学习100回合
        observation = env.reset()   # 初始化state的观测值，即左上角的格子位置，为[5.0, 5.0, 35.0, 35.0]，每个回合都要从这个格子重新开始
        
        # 不断移动机器人，直到下地狱/上天堂（暂将动作执行者称为机器人，在此即RL）
        while True:     
            env.render()    # 更新可视化环境（即能看清每一步是怎么走的

            action = RL.choose_action(str(observation))     # RL机器人根据观测值（即位置）选择动作
            observation_, reward, done = env.step(action)   # RL机器人执行动作，并返回新观测值即位置、奖励及是否结束的标记（下地狱或上天堂均结束）
            RL.learn(str(observation), action, reward, str(observation_))   # RL机器人根据‘新旧观测值、行为、奖励’来学习
           
            # 打印相关信息，以便更好理解
            print("选取动作：\n", action)
            print("观测值，即本位置：\n", observation)
            print("新观测值，即下一位置：\n", observation_)
            print("奖励：\n", reward)
            print("done？：\n", done)
            print("Q表：\n", RL.q_table)
            print("-" * 50 + "\n")

            observation = observation_      # 将新观测值作为下一次的初始观测值

            # 若下地狱或上天堂，则结束循环
            if done:
                break

        print("~" * 50)
        print("第%d回合结束" %(episode+1))
        print("~" * 50 + "\n")


    # 游戏结束，销毁环境
    print('Game Over')
    env.destroy()
    RL.q_table.to_csv('./model.csv')      # 【可选】将最终的Q表存储下来，Q表即训练之后的模型；（本想将模型保存下来以便下次直接使用，但pandas的存储与读取遇到小问题，非重点，故暂搁置）

if __name__ == "__main__":
    env = Maze()    # 创建迷宫环境
    RL = QLearningTable(actions=list(range(env.n_actions)))     # 声明RL，即强化学习的行动者，暂称机器人；参数actions=[0, 1, 2, 3]
    env.after(100, update)      # tkinter的after函数；每100ms调用一次update函数
    env.mainloop()      # 启动tkinter，即以窗口形式显示环境

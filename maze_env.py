"""
Reinforcement learning maze example.

Red rectangle:          explorer.
Black rectangles:       hells       [reward = -1].
Yellow bin circle:      paradise    [reward = +1].
All other states:       ground      [reward = 0].

This script is the environment part of this example. The RL is in RL_brain.py.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
"""

import time
import sys
import numpy as np
import tkinter as tk

UNIT = 40   # 每次移动格子的单位
MAZE_H = 4  # grid height
MAZE_W = 4  # grid width

class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']    # 定义动作空间，上下左右
        self.n_actions = len(self.action_space)
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell
        hell1_center = origin + np.array([UNIT * 2, UNIT])
        self.hell1 = self.canvas.create_rectangle(
            hell1_center[0] - 15, hell1_center[1] - 15,
            hell1_center[0] + 15, hell1_center[1] + 15,
            fill='black')
        # hell
        hell2_center = origin + np.array([UNIT, UNIT * 2])
        self.hell2 = self.canvas.create_rectangle(
            hell2_center[0] - 15, hell2_center[1] - 15,
            hell2_center[0] + 15, hell2_center[1] + 15,
            fill='black')

        # create oval
        oval_center = origin + UNIT * 2
        self.oval = self.canvas.create_oval(
            oval_center[0] - 15, oval_center[1] - 15,
            oval_center[0] + 15, oval_center[1] + 15,
            fill='yellow')

        # create red rect
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')

        # pack all
        self.canvas.pack()

    def reset(self):
        # 返回初始的观测值，左上角第一个格子
        # 观测值，表示位置，用“左上角和右下角两个点的坐标”表示，如[x1,y1,x2,y2]
        self.update()
        time.sleep(0.5)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 15, origin[1] - 15,
            origin[0] + 15, origin[1] + 15,
            fill='red')
        return self.canvas.coords(self.rect)

    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])

        # 根据动作，相应的上下左右移动
        if action == 0:   # 上
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # 下
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # 左
            if s[0] > UNIT:
                base_action[0] -= UNIT
        elif action == 3:   # 右
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT       

        self.canvas.move(self.rect, base_action[0], base_action[1])  # 移动，进入下一格子
        s_ = self.canvas.coords(self.rect)  # 下一个状态/状态的观测值，二者同义

        # 如何返回奖励？天堂或地狱，state的观测值（即位置）均用terminal表示，而不用四元素列表表示
        if s_ == self.canvas.coords(self.oval):          # 天堂，奖励为1，位置标记为terminal即终点
            reward = 1
            done = True
            s_ = 'terminal'        
        elif s_ in [self.canvas.coords(self.hell1), self.canvas.coords(self.hell2)]:    # 地狱，奖励为-1，同样，位置标记为terminal即终点
            reward = -1
            done = True
            s_ = 'terminal'   
        else:   # 普通位置，无奖励，无结束标记
            reward = 0
            done = False
        # 返回新状态、奖励、是否结束的标记
        return s_, reward, done

    def render(self):
        time.sleep(0.1)     # 延迟执行t，以s为单位
        self.update()       # ？update方法定义在类外，为何能self调用 ？


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
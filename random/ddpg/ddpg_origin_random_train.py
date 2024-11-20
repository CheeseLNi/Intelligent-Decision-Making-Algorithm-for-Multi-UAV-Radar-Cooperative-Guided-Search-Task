from stable_baselines3 import DDPG
import torch
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch.nn import functional as F
import gym
from gym import spaces
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_find_nums = open("G:/decrease_model/random/ddpg_origin/ddpg_origin_random_findnums.csv", "w+")

class xietong_bu0_env(gym.Env):
    """A kongyu huafen environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(xietong_bu0_env, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(6,), dtype=np.float16)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(60,), dtype=np.float16)

        self.target_num = 30  # 目标数目
        self.xietong_num = 3    # 协同数目
        self.search_range_x = [10, 20, 30]  # 方位搜索范围
        self.search_range_y = [10, 20, 30]  # 俯仰搜索范围

        self._max_episode_steps = 10    # 最大搜索10次
        self.t = 0  # 全局执行步长

    def reset(self):
        self.x_m = np.random.uniform(-60, 60, size=(self.target_num,))
        self.y_m = np.random.uniform(-60, 60, size=(self.target_num,))
        self.target_cover_situation = np.zeros((self.target_num,))
        # self.obs = np.hstack((self.target_cover_situation, self.x_m, self.y_m))  # taget_num * 3
        self.global_target_uncover = []
        for i in range(self.target_num):
            if self.target_cover_situation[i] == 0:  # 说明目标未被覆盖
                target_bh = int(i + 1)
                self.global_target_uncover.append(target_bh)

        self.steps = 0
        self.rl_center_x = []
        self.rl_center_y = []
        self.greedy_center_x = []
        self.greedy_center_y = []
        self.alpha = 1.0    # 与贪心算法差值奖励
        self.beta = 1.0     # 本次搜索空域发现目标奖励
        # self.gamma = 1.0 / (self.scan_range_x * self.scan_range_y)  # 空域冗余度，直接看差几个空域
        self.t0 = 5e5   # 1e5步差不多才考虑冗余度影响
        self.r0 = 20.0    # 完成任务奖励
        self.rl_num = 0     # 强化学习完成任务需要的空域数

        obs = np.zeros((1, 2 * self.target_num))
        for i in range(np.size(self.global_target_uncover)):
            # obs[0, 2 * i] = int(self.x_m[i] + 60.0)
            # obs[0, 2 * i + 1] = int(self.y_m[i] + 60.0)  # 状态归一化
            obs[0, 2 * i] = self.x_m[i] / 60.0
            obs[0, 2 * i + 1] = self.y_m[i] / 60.0

        return obs

    def step(self, action):
        self.t += 1     # 全局执行步长，不会随着reset归零
        self.rl_num += 1
        self.scan_center_x = [action[0] * 60.0, action[2] * 60.0, action[4] * 60.0]
        self.scan_center_y = [action[1] * 60.0, action[3] * 60.0, action[5] * 60.0]

        target_num_start = np.size(self.global_target_uncover)

        for i in range(self.target_num):
            for j in range(self.xietong_num):
                if abs(self.x_m[i] - self.scan_center_x[j]) < self.search_range_x[j] / 2 and \
                    abs(self.y_m[i] - self.scan_center_y[j]) < self.search_range_y[j] / 2:
                    self.target_cover_situation[i] += 1
                    if self.global_target_uncover.count(i + 1) >= 1:    # 判断这个目标是否被发现
                        self.global_target_uncover.remove(i + 1)    # 从未发现目标集合中剔除出去
        # 遍历完之后直接判断是否完成
        if np.size(self.global_target_uncover) > 0:
            done = 0
        else:
            done = 1

        target_num_end = np.size(self.global_target_uncover)

        # 各个奖励初始化
        reward5 = -1
        reward1, reward2, reward3, reward4 = 0.0, 0.0, 0.0, 0.0     # 只有reward2,reward4,reward5起作用
        reward2 = (target_num_start - target_num_end) * self.beta  # 发现目标数

        if int(done):
            reward4 = self.r0    # 完成回合奖励加50

        self.steps += 1
        if self.steps >= self._max_episode_steps and int(done) == 0:    # 到达最大步数且未完成
            reward4 = -self.r0   # 没完成减50
            done = 1

        if int(done):
            '''保存本回合发现目标数'''
            log_find_nums.write('{}\n'.format(self.target_num - np.size(self.global_target_uncover)))
            log_find_nums.flush()

        reward = reward1 + reward2 + reward3 + reward4 + reward5
        # reward = reward2

        obs = np.zeros((1, 2 * self.target_num))
        for i in range(np.size(self.global_target_uncover)):
            target_bh = int(self.global_target_uncover[i] - 1)
            # obs[0, 2 * i] = int(self.x_m[target_bh] + 60.0)
            # obs[0, 2 * i + 1] = int(self.y_m[target_bh] + 60.0)  # 状态归一化
            obs[0, 2 * i] = self.x_m[target_bh] / 60.0
            obs[0, 2 * i + 1] = self.y_m[target_bh] / 60.0

        # return obs.reshape(1, 30), [reward], [int(done)], [{}]
        return obs, reward, int(done), {}

    def env_render(self):
        plt.plot(self.x_m, self.y_m, 'k*')
        plt.axis('equal')
        plt.show()

# 创建一个SaveCallback，用于在训练过程中保存模型
def save_model_callback(_locals, _globals):
    # 获取当前训练步数
    current_step = _locals["self"].num_timesteps
    save_interval = 10000  # 每10000步保存一次模型
    if current_step % save_interval == 0:
        # 设置模型保存路径和文件名
        model_save_path = f"G:/decrease_model/saved_model/ddpg/ddpg_origin/ddpg_origin_random_{current_step:.1e}.zip"
        # 保存模型
        _locals["self"].save(model_save_path)
        # print(f"Model saved at step {current_step}.")

env = xietong_bu0_env()

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
model = DDPG("MlpPolicy", env=env, gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="G:/decrease_model/result/ddpg_origin_random", verbose=1, device=device)
model.learn(total_timesteps=int(1e6), callback=save_model_callback, log_interval=50)
model.save("ddpg_origin_random.zip")
















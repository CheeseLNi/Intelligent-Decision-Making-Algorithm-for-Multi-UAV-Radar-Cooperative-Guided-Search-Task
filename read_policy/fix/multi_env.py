import gym
from gym import spaces
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class td3_bu0_env(gym.Env):
    """A kongyu huafen environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(td3_bu0_env, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(6,), dtype=np.float16)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(60,), dtype=np.float16)

        self.target_num = 30  # 目标数目
        self.xietong_num = 3    # 协同数目
        self.search_range_x = [10, 20, 30]  # 方位搜索范围
        self.search_range_y = [10, 20, 30]  # 俯仰搜索范围

        self.x_m = np.array([-42.67478786, -19.47873314, 52.20419535, 18.82398507, -52.42192912,
                            -50.57795674, -15.88859449, -8.85018843, 2.11457904, 21.78440157,
                            35.46206604, 13.30316108, -29.90699771, -30.47413791, 32.18445288,
                            41.42905077, 28.21599124, 51.43963535, -56.69589378, 19.63291292,
                            -51.34703292, 6.35275373, -15.02784673, -37.14532043, 41.53851406,
                            -5.09526259, 17.79560894, 15.11512674, -0.71066349, 20.66251949])
        self.y_m = np.array([54.30130362, -18.89145101, 5.10122446, -48.96692001, -2.07999212,
                            -14.39694055, -3.26693532, -52.08329969, -55.36285795, -14.27684586,
                            -33.12414592, -0.74059242, 31.23614159, -7.0675359, 3.30129922,
                            -51.68269094, 26.91533069, 1.12160894, -47.466137, -51.74942182,
                            9.30348708, 3.48665783, -18.17954256, -1.89079884, 20.38351896,
                            -45.68298136, 0.2185957, -51.41024967, -1.39676394, 17.73277371])
        self._max_episode_steps = 10    # 最大搜索10次
        self.t = 0  # 全局执行步长

    def reset(self):
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

        # if int(done):
        #     '''保存本回合发现目标数'''
        #     log_find_nums.write('{}\n'.format(self.target_num - np.size(self.global_target_uncover)))
        #     log_find_nums.flush()

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

class ddpg_bu0_env(gym.Env):
    """A kongyu huafen environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(ddpg_bu0_env, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(6,), dtype=np.float16)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(60,), dtype=np.float16)

        self.target_num = 30  # 目标数目
        self.xietong_num = 3    # 协同数目
        self.search_range_x = [10, 20, 30]  # 方位搜索范围
        self.search_range_y = [10, 20, 30]  # 俯仰搜索范围

        self.x_m = np.array([-42.67478786, -19.47873314, 52.20419535, 18.82398507, -52.42192912,
                            -50.57795674, -15.88859449, -8.85018843, 2.11457904, 21.78440157,
                            35.46206604, 13.30316108, -29.90699771, -30.47413791, 32.18445288,
                            41.42905077, 28.21599124, 51.43963535, -56.69589378, 19.63291292,
                            -51.34703292, 6.35275373, -15.02784673, -37.14532043, 41.53851406,
                            -5.09526259, 17.79560894, 15.11512674, -0.71066349, 20.66251949])
        self.y_m = np.array([54.30130362, -18.89145101, 5.10122446, -48.96692001, -2.07999212,
                            -14.39694055, -3.26693532, -52.08329969, -55.36285795, -14.27684586,
                            -33.12414592, -0.74059242, 31.23614159, -7.0675359, 3.30129922,
                            -51.68269094, 26.91533069, 1.12160894, -47.466137, -51.74942182,
                            9.30348708, 3.48665783, -18.17954256, -1.89079884, 20.38351896,
                            -45.68298136, 0.2185957, -51.41024967, -1.39676394, 17.73277371])
        self._max_episode_steps = 10    # 最大搜索10次
        self.t = 0  # 全局执行步长

    def reset(self):
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

        # if int(done):
        #     '''保存本回合发现目标数'''
        #     log_find_nums.write('{}\n'.format(self.target_num - np.size(self.global_target_uncover)))
        #     log_find_nums.flush()

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

class maddpg_bu0_env(gym.Env):
    """A kongyu huafen environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(maddpg_bu0_env, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(2,), dtype=np.float16)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(60,), dtype=np.float16)

        self.target_num = 30  # 目标数目
        self.xietong_num = 1    # 协同数目
        self.search_range_x = [10, 20, 30]  # 方位搜索范围
        self.search_range_y = [10, 20, 30]  # 俯仰搜索范围

        self.x_m = np.array([-42.67478786, -19.47873314, 52.20419535, 18.82398507, -52.42192912,
                            -50.57795674, -15.88859449, -8.85018843, 2.11457904, 21.78440157,
                            35.46206604, 13.30316108, -29.90699771, -30.47413791, 32.18445288,
                            41.42905077, 28.21599124, 51.43963535, -56.69589378, 19.63291292,
                            -51.34703292, 6.35275373, -15.02784673, -37.14532043, 41.53851406,
                            -5.09526259, 17.79560894, 15.11512674, -0.71066349, 20.66251949])
        self.y_m = np.array([54.30130362, -18.89145101, 5.10122446, -48.96692001, -2.07999212,
                            -14.39694055, -3.26693532, -52.08329969, -55.36285795, -14.27684586,
                            -33.12414592, -0.74059242, 31.23614159, -7.0675359, 3.30129922,
                            -51.68269094, 26.91533069, 1.12160894, -47.466137, -51.74942182,
                            9.30348708, 3.48665783, -18.17954256, -1.89079884, 20.38351896,
                            -45.68298136, 0.2185957, -51.41024967, -1.39676394, 17.73277371])
        self._max_episode_steps = 30    # 单个最大搜索是30次
        self.t = 0  # 全局执行步长

    def reset(self):
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
        self.scan_center_x = [action[0] * 60.0]
        self.scan_center_y = [action[1] * 60.0]
        agent_no = action[2]

        target_num_start = np.size(self.global_target_uncover)

        # 不应该只遍历未发现目标，这样还是会导致不发生变化
        for i in range(self.target_num):
            for j in range(self.xietong_num):
                if abs(self.x_m[i] - self.scan_center_x[j]) < self.search_range_x[agent_no] / 2 and \
                    abs(self.y_m[i] - self.scan_center_y[j]) < self.search_range_y[agent_no] / 2:
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

        # if int(done):
        #     '''保存本回合发现目标数'''
        #     log_find_nums.write('{}\n'.format(self.target_num - np.size(self.global_target_uncover)))
        #     log_find_nums.flush()

        reward4 = 0.0
        reward = reward1 + reward2 + reward3 + reward4 + reward5
        # reward = reward2

        obs = np.zeros((1, 2 * self.target_num))
        for i in range(np.size(self.global_target_uncover)):
            target_bh = int(self.global_target_uncover[i] - 1)
            # obs[0, 2 * i] = int(self.x_m[target_bh] + 60.0)
            # obs[0, 2 * i + 1] = int(self.y_m[target_bh] + 60.0)  # 状态归一化
            obs[0, 2 * i] = self.x_m[target_bh] / 60.0
            obs[0, 2 * i + 1] = self.y_m[target_bh] / 60.0

        return obs, [reward], [int(done)], [{}]

    def env_render(self):
        plt.plot(self.x_m, self.y_m, 'k*')
        plt.axis('equal')
        plt.show()

# 这个环境仅仅用来更新策略网络，也就是说Critic网络的输入就是state_dim + 3 * action_dim = 30 + 6 = 36
class maddpg_xietong_env(gym.Env):
    """A kongyu huafen environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(maddpg_xietong_env, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(6,), dtype=np.float16)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(60,), dtype=np.float16)

    def reset(self):
        return np.zeros((1, 60))






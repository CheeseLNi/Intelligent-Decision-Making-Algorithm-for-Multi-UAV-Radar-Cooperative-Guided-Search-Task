from stable_baselines3 import DDPG
import torch
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch.nn import functional as F
from stable_baselines3.common.utils import polyak_update
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.utils import safe_mean, should_collect_more_steps
import gym
from gym import spaces
import threading
from copy import deepcopy
import matplotlib.pyplot as plt
import torch.nn as nn
torch.backends.cudnn.enabled = False  # 关闭cuDNN加速
log_find_nums = open("D:/decrease_model/fix/matd3/matd3_fix_findnums.csv", "w+")
log_ep_rew_mean = open("D:/decrease_model/fix/matd3/matd3_fix_ep_rew_mean.csv", "w+")
log_ep_len_mean = open("D:/decrease_model/fix/matd3/matd3_fix_ep_len_mean.csv", "w+")

class single_bu0_env(gym.Env):
    """A kongyu huafen environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(single_bu0_env, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(60,), dtype=np.float32)

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

class single_bu0_eval_env(gym.Env):
    """A kongyu huafen environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(single_bu0_eval_env, self).__init__()

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

        if int(done):
            '''保存本回合发现目标数'''
            log_find_nums.write('{}\n'.format(self.target_num - np.size(self.global_target_uncover)))
            log_find_nums.flush()

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
class xietong_env(gym.Env):
    """A kongyu huafen environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(xietong_env, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(6,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(60,), dtype=np.float32)

    def reset(self):
        return np.zeros((1, 60))


n_embd = 16
block_size = 4
dropout = 0.5
n_head = 4
n_layer = 1
n_seq = n_head

class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B, T, C = x.shape
        k = self.key(x)  # (B,T,hs)
        q = self.query(x)  # (B,T,hs)
        # compute attention scores ("affinities")
        wei = (
            q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5
        )  # (B, T, hs) @ (B, hs, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))  # (B, T, T)
        wei = F.softmax(wei, dim=-1)  # (B, T, T)
        wei = self.dropout(wei)
        # perform the weighted aggregation of the values
        v = self.value(x)  # (B,T,hs)
        out = wei @ v  # (B, T, T) @ (B, T, hs) -> (B, T, hs)
        return out

class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)     # 在给定维度上对输入的张量序列seq 进行连接操作
        out = self.dropout(self.proj(out))
        return out

class FeedFoward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd),
            nn.ReLU(),
            nn.Linear(n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head    # 整数除法？
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedFoward(n_embd)  # n_embd, 4 * n_embd, n_embd
        self.ln1 = nn.LayerNorm(n_embd)     # 将指定的数据归一化，即均值为0，方差为1
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))    # 输入x：torch.Size([64, 256, 384])，self.ln1(x)：torch.Size([64, 256, 384])，self.sa(self.ln1(x))：torch.Size([64, 256, 384])
        x = x + self.ffwd(self.ln2(x))  # self.ffwd(self.ln2(x))：torch.Size([64, 256, 384])
        return x

class att_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.my_embd1 = nn.Linear(in_features=60, out_features=n_embd * n_head)
        # self.my_embd2 = nn.Linear(in_features=64 * 4, out_features=64 * 16)
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(in_features=n_embd * n_head, out_features=n_embd)
        self.fc2 = nn.Linear(in_features=n_embd, out_features=2)
        self.lstm = nn.LSTM(n_embd, n_embd, n_layer, batch_first=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.blocks = nn.Sequential(
            *[
                Block(n_embd, n_head)
                for _ in range(n_layer)
            ]
        )
        self.apply(self._init_weights)  # 权值初始化

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # print(idx.shape)
        # x = self.embedding(x.long())   # (batch_size,state_dim)->(batch_size,state_dim,n_embd)
        x = self.relu(self.my_embd1(x))
        # x = self.relu(self.my_embd2(x))
        x = x.reshape(x.size(0), n_head, n_embd)
        # x, _ = self.lstm(x)
        x = self.blocks(x)
        # x = self.blocks(x)      # (batch_size,state_dim,n_embd)->(batch_size,state_dim,n_embd)
        x = self.relu(self.fc1(x.reshape(x.size(0), -1)))   # (batch_size,state_dim,n_embd)->(batch_size,state_dim*n_embd)
        x = self.tanh(self.fc2(x))  # (batch_size,state_dim*n_embd)->(batch_size,action_dim)
        return x

# 在模型定义部分增加双Critic网络
class MATD3Critic(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        # Q1 architecture
        self.q1_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # Q2 architecture
        self.q2_net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        # 独立优化器
        # self.optimizer_q1 = None
        # self.optimizer_q2 = None
        # 统一优化器接口
        self.optimizers = [
            torch.optim.Adam(self.q1_net.parameters(), lr=2e-5),
            torch.optim.Adam(self.q2_net.parameters(), lr=2e-5)
        ]

    def forward(self, obs, act):
        x = torch.cat([obs, act], dim=-1)
        return self.q1_net(x), self.q2_net(x)

def collect_xietong_rollouts(multi_agent, xietong_agent, multi_callback, xietong_callback):
    # 以协同评判智能体为基准，其它智能体只搜集各自的轨迹，然后使用总的轨迹更新全协同评判智能体
    xietong_agent.policy.set_training_mode(False)
    Agent_num = np.size(multi_agent)
    for i in range(Agent_num):
        multi_agent[i].policy.set_training_mode(False)

    num_collected_steps, num_collected_episodes = 0, 0  # 共用这两个参数

    for i in range(Agent_num):  # 只需要初始化三个动作智能体的动作噪声
        # Vectorize action noise if needed
        if multi_agent[i].action_noise is not None and multi_agent[i].env.num_envs > 1 and not isinstance(multi_agent[i].action_noise, VectorizedActionNoise):
            multi_agent[i].action_noise = VectorizedActionNoise(multi_agent[i].action_noise, multi_agent[i].env.num_envs)
        if multi_agent[i].use_sde:
            multi_agent[i].actor.reset_noise(multi_agent[i].env.num_envs)

    xietong_callback.on_rollout_start()
    for i in range(Agent_num):
        multi_callback[i].on_rollout_start()
    continue_training = True

    single_agent_collect_env = single_bu0_env()
    new_obs = single_agent_collect_env.reset()
    multi_agent[0]._last_obs = deepcopy(new_obs)  # 这里设置第一个智能体的初始状态，后续依据该智能体动作依次更新其它两个智能体状态

    # ep_rew = 0.0

    while should_collect_more_steps(xietong_agent.train_freq, num_collected_steps, num_collected_episodes):     # 这里用协同智能体进行回合判断
        for i in range(Agent_num):  # 仍然只更新动作智能体的动作噪声
            if multi_agent[i].use_sde and multi_agent[i].sde_sample_freq > 0 and num_collected_steps % multi_agent[i].sde_sample_freq == 0:
                # Sample a new noise matrix
                multi_agent[i].actor.reset_noise(multi_agent[i].env.num_envs)

        # Select action randomly or according to policy，直接在这里收集轨迹了
        multi_actions = []
        multi_buffer_actions = []
        multi_rewards = 0.0
        current_obs = deepcopy(new_obs)     # 协同轨迹起始状态
        for i in range(Agent_num):
            # 选择动作
            if xietong_agent.num_timesteps < xietong_agent.learning_starts and not (xietong_agent.use_sde and xietong_agent.use_sde_at_warmup): # 起始阶段，随机采样
                unscaled_action = np.array([multi_agent[i].action_space.sample() for _ in range(multi_agent[i].n_envs)])
            else:
                unscaled_action, _ = multi_agent[i].predict(new_obs, deterministic=False)
            scaled_action = multi_agent[i].policy.scale_action(unscaled_action)
            if multi_agent[i].action_noise is not None:
                scaled_action = np.clip(scaled_action + multi_agent[i].action_noise(), -1, 1)
            # We store the scaled action in the buffer
            buffer_actions = scaled_action
            actions = multi_agent[i].policy.unscale_action(scaled_action)

            multi_actions.append(deepcopy(actions))
            multi_buffer_actions.append(deepcopy(buffer_actions))
            # 执行动作
            new_obs, rewards, dones, infos = single_agent_collect_env.step([actions[0][0], actions[0][1], i])
            multi_rewards += deepcopy(rewards[0])
            multi_agent[(i + 1) % Agent_num]._last_obs = deepcopy(current_obs)
        next_obs = deepcopy(new_obs)    # 协同轨迹执行完协同动作后的状态
        if dones[0]:  # 到达最大步数且未完成
            # 直接判断状态空间第一个值，判断是否为零
            if abs(next_obs[0, 0]) > 1e-6:  # 状态空间不是全零，说明有目标未发现
                multi_rewards += -20.0
            else:
                multi_rewards += 20.0
        # if dones[0]:    # 到达最大步数且未完成
        #     for i in range(30):
        #         if next_obs[0, i] == 0:     # 任意一个目标被发现的次数为0，说明任务没完成，直接break
        #             multi_rewards += -20.0
        #             break
        #         else:
        #             if i == 29:  # 一直到29都没break，并且next_obs[0, i] != 0，则说明任务完成
        #                 multi_rewards += 20.0
        multi_rewards += Agent_num - 1  # 这里相当于减了Agent_num个1，所以加回来
        # ep_rew += multi_rewards
        multi_buffer_actions = np.hstack(multi_buffer_actions)

        xietong_agent.num_timesteps += xietong_agent.env.num_envs
        num_collected_steps += 1

        # if dones[0]:
        #     log_ep_len_mean.write('{}\n'.format(num_collected_steps))
        #     log_ep_len_mean.flush()
        #     log_ep_rew_mean.write('{}\n'.format(ep_rew))
        #     log_ep_rew_mean.flush()

        # Give access to local variables
        xietong_callback.update_locals(locals())
        # Only stop training if return value is False, not when it is None.
        if xietong_callback.on_step() is False:
            return RolloutReturn(num_collected_steps * xietong_agent.env.num_envs, num_collected_episodes, continue_training=False)

        xietong_agent._update_info_buffer(infos, dones)
        xietong_agent.replay_buffer.add(deepcopy(current_obs), deepcopy(next_obs), deepcopy(multi_buffer_actions), deepcopy([multi_rewards]), deepcopy(dones), deepcopy(infos), )
        xietong_agent._update_current_progress_remaining(xietong_agent.num_timesteps, xietong_agent._total_timesteps)
        xietong_agent._on_step()

        for idx, done in enumerate(dones):
            if done:
                # Update stats
                num_collected_episodes += 1
                xietong_agent._episode_num += 1
                for i in range(Agent_num):
                    if multi_agent[i].action_noise is not None:
                        kwargs = dict(indices=[idx]) if multi_agent[i].env.num_envs > 1 else {}
                        multi_agent[i].action_noise.reset(**kwargs)

    xietong_callback.on_rollout_end()
    for i in range(Agent_num):
        multi_callback[i].on_rollout_end()
    return RolloutReturn(num_collected_steps * xietong_agent.env.num_envs, num_collected_episodes, continue_training)

# 修改训练函数
def train_agent(multi_agent, xietong_agent, gradient_steps):
    Agent_num = np.size(multi_agent)
    policy_delay = 2  # 延迟策略更新参数
    # 初始化优化器（仅在首次训练时）
    if not hasattr(xietong_agent.critic, 'optimizers'):
        xietong_agent.critic.optimizers = [
            torch.optim.Adam(xietong_agent.critic.q1_net.parameters(), lr=xietong_agent.learning_rate),
            torch.optim.Adam(xietong_agent.critic.q2_net.parameters(), lr=xietong_agent.learning_rate)
        ]
    for _ in range(gradient_steps):
        xietong_agent._n_updates += 1
        
        # 从经验池采样
        replay_data = xietong_agent.replay_buffer.sample(batch_size, env=xietong_agent._vec_normalize_env)

        with torch.no_grad():
            # 目标策略平滑
            noise = (torch.randn_like(replay_data.actions) * xietong_agent.target_policy_noise).clamp(
                -xietong_agent.target_noise_clip, xietong_agent.target_noise_clip
            )
            
            # 计算目标动作
            multi_next_actions = []
            for i in range(Agent_num):
                next_actions = multi_agent[i].actor_target(replay_data.next_observations)
                multi_next_actions.append(next_actions)
            multi_next_actions = torch.cat(multi_next_actions, dim=1)
            noisy_next_actions = (multi_next_actions + noise).clamp(-1, 1)

            # 计算双Critic目标Q值
            target_Q1, target_Q2 = xietong_agent.critic_target(replay_data.next_observations, noisy_next_actions)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = replay_data.rewards + (1 - replay_data.dones) * xietong_agent.gamma * target_Q

        # 计算Critic损失
        current_Q1, current_Q2 = xietong_agent.critic(replay_data.observations, replay_data.actions)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        optimizer_q1 = xietong_agent.critic.optimizers[0]
        optimizer_q2 = xietong_agent.critic.optimizers[1]
        # 分别优化两个Critic
        optimizer_q1.zero_grad()
        optimizer_q2.zero_grad()
        critic_loss.backward()
        optimizer_q1.step()
        optimizer_q2.step()

        # 延迟策略更新
        if xietong_agent._n_updates % policy_delay == 0:
            # 更新Actor
            multi_actor_loss = []
            for i in range(Agent_num):
                # 重新生成动作
                action_rechoose = multi_agent[i].actor(replay_data.observations)
                new_actions = replay_data.actions.clone()
                new_actions[:, 2*i:2*(i+1)] = action_rechoose
                
                # 使用Q1计算策略梯度
                actor_q_value, _ = xietong_agent.critic(replay_data.observations, new_actions)
                actor_loss = -actor_q_value.mean()
                
                multi_agent[i].actor.optimizer.zero_grad()
                actor_loss.backward(retain_graph=True)
                multi_agent[i].actor.optimizer.step()
                multi_actor_loss.append(actor_loss.item())

            # 更新目标网络
            with torch.no_grad():
                polyak_update(xietong_agent.critic.parameters(), xietong_agent.critic_target.parameters(), xietong_agent.tau)
                for i in range(Agent_num):
                    polyak_update(multi_agent[i].actor.parameters(), multi_agent[i].actor_target.parameters(), multi_agent[i].tau)

def evaluate_policy(multi_agent):
    Agent_num = np.size(multi_agent)
    single_agent_collect_env = single_bu0_eval_env()
    multi_actions = []
    reward = 0
    obs = single_agent_collect_env.reset()
    dones = [0]
    steps = 0
    while not int(dones[0]):
        steps += 1
        for i in range(Agent_num):
            actions = multi_agent[i].predict(obs)
            actions_reshape = np.zeros((2,))
            actions_reshape[0], actions_reshape[1] = actions[0][0][0], actions[0][0][1]
            # multi_actions.append(deepcopy(actions_reshape)),
            obs, rewards, dones, infos = single_agent_collect_env.step([actions_reshape[0], actions_reshape[1], i])
            reward += rewards[0]
        reward += Agent_num - 1.0
        if dones[0]:  # 到达最大步数且未完成
            # 直接判断状态空间第一个值，判断是否为零
            if abs(obs[0, 0]) > 1e-6:  # 状态空间不是全零，说明有目标未发现
                reward += -20.0
            else:
                reward += 20.0
            log_ep_len_mean.write('{}\n'.format(steps))
            log_ep_len_mean.flush()
            log_ep_rew_mean.write('{}\n'.format(reward))
            log_ep_rew_mean.flush()
            return reward
        # if dones[0]:  # 到达最大步数且未完成
        #     for i in range(30):
        #         if obs[0, i] == 0:  # 任意一个目标被发现的次数为0，说明任务没完成，直接break
        #             reward += -20.0
        #             break
        #         else:
        #             if i == 29:  # 一直到29都没break，并且next_obs[0, i] != 0，则说明任务完成
        #                 reward += 20.0
        #     return reward

# 修改智能体初始化部分
class MATD3Agent(DDPG):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        # 替换原始Critic为双Critic
        self.critic = MATD3Critic(self.observation_space.shape[0], self.action_space.shape[0]).to(self.device)
        self.critic_target = deepcopy(self.critic)
        # 禁用基类自动保存优化器
        self._excluded_save_params = ["optimizer"]
    def save(self, path: str) -> None:
        """自定义保存方法"""
        save_dict = {
            "policy_state": self.policy.state_dict(),
            "critic_state": self.critic.state_dict(),
            "critic_target_state": self.critic_target.state_dict(),
            "critic_optimizers": [
                self.critic.optimizers[0].state_dict(),
                self.critic.optimizers[1].state_dict()
            ]
        }
        torch.save(save_dict, path)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
n_actions = single_bu0_env().action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

att_test_model = att_model().to(device)
# 前三个智能体的动作空间均为2，状态空间均为30
agent1 = DDPG("MlpPolicy", env=single_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="D:\\decrease_model\\result\\2025_2_27\\xietong_result", verbose=1, device=device)
agent2 = DDPG("MlpPolicy", env=single_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="D:\\decrease_model\\result\\2025_2_27\\xietong_result", verbose=1, device=device)
agent3 = DDPG("MlpPolicy", env=single_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="D:\\decrease_model\\result\\2025_2_27\\xietong_result", verbose=1, device=device)
# multi_agent = [agent1, agent2, agent3]  # 三个智能体只用策略网络
multi_agent = [agent1, agent2, agent3]
agent_num = np.size(multi_agent)
for i in range(agent_num): 
    # multi_agent[i].policy.actor.mu = deepcopy(att_test_model)
    # multi_agent[i].policy.actor_target.mu = deepcopy(att_test_model)
    multi_agent[i].policy.actor_target.load_state_dict(multi_agent[i].actor.state_dict())
    multi_agent[i].policy.actor.optimizer = multi_agent[i].policy.optimizer_class(multi_agent[i].policy.actor.parameters())
# 协同智能体动作空间为6，状态空间为30
n_actions = xietong_env().action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
# 初始化协同智能体时使用MATD3Agent
xietong_agent = MATD3Agent("MlpPolicy", env=xietong_env(), gamma=0.98, 
                          policy_kwargs=policy_kwargs, learning_rate=0.0001,
                          tensorboard_log="D:\\decrease_model\\result\\2025_2_27\\xietong_result", device=device)
# 在智能体初始化参数中增加
xietong_agent.target_policy_noise = 0.2      # 目标策略噪声
xietong_agent.target_noise_clip = 0.5        # 噪声裁剪范围
xietong_agent.policy_delay = 2               # 策略延迟更新参数
# 超参数
gradient_steps = 10
batch_size = 100

total_timesteps = int(1e6)
callback = None
log_interval = 4
eval_env = None
eval_freq = -1
n_eval_episodes = 5
tb_log_name = 'MATD3'
eval_log_path = None
reset_num_timesteps = True
progress_bar = False

multi_total_timesteps = []
multi_callback = []
multi_rollout = []

# 大循环
reward_avg = 0.0
step = 0

# multi_agent[0].load("maddpg_att_fix_actor1_1e6.zip")
# multi_agent[1].load("maddpg_att_fix_actor2_1e6.zip")
# multi_agent[2].load("maddpg_att_fix_actor3_1e6.zip")
# xietong_agent.load("maddpg_att_fix_critic_1e6.zip")

for i in range(agent_num):
    # total_timesteps, callback = multi_agent[i]._setup_learn(total_timesteps, eval_env, callback, eval_freq,
    #                                                         n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name)
    total_timesteps, callback = multi_agent[i]._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar,)
    callback.on_training_start(locals(), globals())
    multi_total_timesteps.append(deepcopy(total_timesteps))
    # multi_callback.append(deepcopy(callback))
    multi_callback.append(callback)


# xietong_total_timesteps, xietong_callback = xietong_agent._setup_learn(total_timesteps, eval_env, callback, eval_freq,
#                                                             n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name)
xietong_total_timesteps, xietong_callback = xietong_agent._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar,)
xietong_callback.on_training_start(locals(), globals())

while xietong_agent.num_timesteps < total_timesteps:   # 三个智能体设置的训练步长是一样的

    step += 1

    xietong_rollout = collect_xietong_rollouts(multi_agent, xietong_agent, multi_callback, xietong_callback)

    if xietong_agent.num_timesteps > 0 and xietong_agent.num_timesteps > xietong_agent.learning_starts:
        # If no `gradient_steps` is specified,
        # do as many gradients steps as steps performed during the rollout
        gradient_steps = xietong_agent.gradient_steps if xietong_agent.gradient_steps >= 0 else xietong_rollout.episode_timesteps
        # Special case when the user passes `gradient_steps=0`
        if gradient_steps > 0:
            train_agent(multi_agent, xietong_agent, gradient_steps)

    if step % 1000 == 0:
        save_path = f"D:/decrease_model/saved_model/matd3/matd3/matd3_fix_critic_{step:.1e}.zip"
        torch.save({
            "actor": xietong_agent.actor.state_dict(),
            "critic": xietong_agent.critic.state_dict(),
            "critic_optimizers": [
                xietong_agent.critic.optimizers[0].state_dict(),
                xietong_agent.critic.optimizers[1].state_dict()
            ]
        }, save_path)
        multi_agent[0].save(f"D:/decrease_model/saved_model/matd3/matd3/matd3_fix_actor1_{step:.1e}.zip")
        multi_agent[1].save(f"D:/decrease_model/saved_model/matd3/matd3/matd3_fix_actor2_{step:.1e}.zip")
        multi_agent[2].save(f"D:/decrease_model/saved_model/matd3/matd3/mtd3g_fix_actor3_{step:.1e}.zip")

    if step % 5 == 0:
        reward_avg += evaluate_policy(multi_agent)
    if step % 100 == 0:
        print(reward_avg / 20, step, xietong_agent.num_timesteps)
        reward_avg = 0.0

multi_agent[0].save("D:/decrease_model/saved_model/matd3/matd3/matd3_fix_actor1")
multi_agent[1].save("D:/decrease_model/saved_model/matd3/matd3/matd3_fix_actor2")
multi_agent[2].save("D:/decrease_model/saved_model/matd3/matd3/matd3_fix_actor3")
save_path = f"D:/decrease_model/saved_model/matd3/matd3/matd3_fix_critic"
torch.save({
    "actor": xietong_agent.actor.state_dict(),
    "critic": xietong_agent.critic.state_dict(),
    "critic_optimizers": [
        xietong_agent.critic.optimizers[0].state_dict(),
        xietong_agent.critic.optimizers[1].state_dict()
    ]
}, save_path)

# agent1.load("maddpg_fix_actor1.zip")
# agent2.load("maddpg_fix_actor2.zip")
# agent3.load("maddpg_fix_actor3.zip")
# xietong_agent.load("maddpg_fix_critic.zip")
# multi_agent = [agent1, agent2, agent3]
#
# eval_rew = evaluate_policy(multi_agent)
# print(eval_rew)





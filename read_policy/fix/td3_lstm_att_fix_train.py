from stable_baselines3 import TD3
import torch
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from torch.nn import functional as F
import gym
from gym import spaces
import torch.nn as nn
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
log_find_nums = open("G:/decrease_model/fix/td3/td3_lstm_att_fix_findnums.csv", "w+")

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

n_embd = 16
block_size = 4
dropout = 0.5
n_head = 4
n_layer = 1

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
        self.fc2 = nn.Linear(in_features=n_embd, out_features=6)
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
        x, _ = self.lstm(x)
        x = self.blocks(x)
        # x = self.blocks(x)      # (batch_size,state_dim,n_embd)->(batch_size,state_dim,n_embd)
        x = self.relu(self.fc1(x.reshape(x.size(0), -1)))   # (batch_size,state_dim,n_embd)->(batch_size,state_dim*n_embd)
        x = self.tanh(self.fc2(x))  # (batch_size,state_dim*n_embd)->(batch_size,action_dim)
        return x

# 创建一个SaveCallback，用于在训练过程中保存模型
def save_model_callback(_locals, _globals):
    # 获取当前训练步数
    current_step = _locals["self"].num_timesteps
    save_interval = 10000  # 每10000步保存一次模型
    if current_step % save_interval == 0:
        # 设置模型保存路径和文件名
        model_save_path = f"G:/decrease_model/saved_model/td3/td3_lstm_att_fix/td3_lstm_att_fix_{current_step:.2e}.zip"
        # 保存模型
        _locals["self"].save(model_save_path)
        # print(f"Model saved at step {current_step}.")

att_test_model = att_model().to(device)
env = xietong_bu0_env()

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
model = TD3("MlpPolicy", env=env, gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="G:/decrease_model/result/td3_lstm_att_fix", verbose=1, device=device)
# model.load("G:/decrease_model/saved_model/td3/td3_fix/td3_fix_9e+03.zip")
model.policy.actor.mu = att_test_model
model.policy.actor_target.mu = att_test_model
model.policy.actor_target.load_state_dict(model.actor.state_dict())
model.policy.actor.optimizer = model.policy.optimizer_class(model.policy.actor.parameters())
model.learn(total_timesteps=int(1e6), callback=save_model_callback, log_interval=20)
model.save("td3_lstm_att_fix.zip")
















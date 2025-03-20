from stable_baselines3 import DDPG
from stable_baselines3 import TD3
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
from multi_env import td3_bu0_env, ddpg_bu0_env, maddpg_bu0_env, maddpg_xietong_env

def plot_airspace(scan_range_x, scan_range_y, airspace_center_x, airspace_center_y, color, label=None, linestyle='-'):
    plt.plot([airspace_center_x - scan_range_x / 2, airspace_center_x + scan_range_x / 2],
            [airspace_center_y - scan_range_y / 2, airspace_center_y - scan_range_y / 2], color=color, linestyle=linestyle, label=label)
    plt.plot([airspace_center_x + scan_range_x / 2, airspace_center_x + scan_range_x / 2],
            [airspace_center_y - scan_range_y / 2, airspace_center_y + scan_range_y / 2], color=color, linestyle=linestyle)
    plt.plot([airspace_center_x + scan_range_x / 2, airspace_center_x - scan_range_x / 2],
            [airspace_center_y + scan_range_y / 2, airspace_center_y + scan_range_y / 2], color=color, linestyle=linestyle)
    plt.plot([airspace_center_x - scan_range_x / 2, airspace_center_x - scan_range_x / 2],
            [airspace_center_y + scan_range_y / 2, airspace_center_y - scan_range_y / 2], color=color, linestyle=linestyle)

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

class single_att_model(nn.Module):
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
        x, _ = self.lstm(x)
        x = self.blocks(x)
        # x = self.blocks(x)      # (batch_size,state_dim,n_embd)->(batch_size,state_dim,n_embd)
        x = self.relu(self.fc1(x.reshape(x.size(0), -1)))   # (batch_size,state_dim,n_embd)->(batch_size,state_dim*n_embd)
        x = self.tanh(self.fc2(x))  # (batch_size,state_dim*n_embd)->(batch_size,action_dim)
        return x

class multi_att_model(nn.Module):
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
td3_lstm_att_model = multi_att_model().to(device)
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
n_actions = td3_bu0_env().action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
td3_model = TD3("MlpPolicy", env=td3_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="G:/decrease_model/result/td3_lstm_att_fix", verbose=1, device=device)
td3_model.policy.actor.mu = td3_lstm_att_model
td3_model.policy.actor_target.mu = td3_lstm_att_model
td3_model.policy.actor_target.load_state_dict(td3_model.actor.state_dict())
td3_model.policy.actor.optimizer = td3_model.policy.optimizer_class(td3_model.policy.actor.parameters())
# td3_model.load("td3_lstm_att_fix_1.00e+06.zip")
td3_policy = torch.load("./td3_lstm_att_fix/policy.pth")
td3_model.policy.load_state_dict(td3_policy)

td3_eval_env = td3_bu0_env()
obs = td3_eval_env.reset()
done = 0
ep_rew = 0
step = 0
action_record = []
while not int(done):
    step += 1
    action, _ = td3_model.predict(obs)
    action = action.reshape(6, -1)
    obs, reward, done, info = td3_eval_env.step(action)
    ep_rew += reward
    action_record.append(action)
print("td3 episode reward = {}".format(ep_rew))
plt.figure(1)
plt.title('td3_lstm_att policy')
plt.plot(td3_eval_env.x_m, td3_eval_env.y_m, 'k^', label='target')
colors = ['red', 'green', 'blue']
for i in range(len(action_record)):
    xietong_scan_center = action_record[i]
    xietong_scan_range = [td3_eval_env.search_range_x, td3_eval_env.search_range_y]
    for j in range(td3_eval_env.xietong_num):
        scan_center_x = xietong_scan_center[j * 2][0] * 60.0
        scan_center_y = xietong_scan_center[j * 2 + 1][0] * 60.0
        scan_range_x = xietong_scan_range[0][j]
        scan_range_y = xietong_scan_range[1][j]
        if i == 0:
            plot_airspace(scan_range_x, scan_range_y, scan_center_x, scan_center_y, colors[j], label='Agent:{}'.format(j), linestyle='-')
        else:
            plot_airspace(scan_range_x, scan_range_y, scan_center_x, scan_center_y, colors[j], linestyle='-')
plt.legend()
plt.axis('equal')
# plt.show()

ddpg_lstm_att_model = multi_att_model().to(device)
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
n_actions = ddpg_bu0_env().action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
ddpg_model = DDPG("MlpPolicy", env=ddpg_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="G:/decrease_model/result/ddpg_lstm_att_fix", verbose=1, device=device)
ddpg_model.policy.actor.mu = ddpg_lstm_att_model
ddpg_model.policy.actor_target.mu = ddpg_lstm_att_model
ddpg_model.policy.actor_target.load_state_dict(ddpg_model.actor.state_dict())
ddpg_model.policy.actor.optimizer = ddpg_model.policy.optimizer_class(ddpg_model.policy.actor.parameters())
# ddpg_model.load("ddpg_lstm_att_fix_1.00e+06.zip")
ddpg_policy = torch.load("./ddpg_lstm_att_fix/policy.pth")
ddpg_model.policy.load_state_dict(ddpg_policy)

ddpg_eval_env = ddpg_bu0_env()
obs = ddpg_eval_env.reset()
done = 0
ep_rew = 0
step = 0
action_record = []
while not int(done):
    step += 1
    action, _ = ddpg_model.predict(obs)
    action = action.reshape(6, -1)
    obs, reward, done, info = ddpg_eval_env.step(action)
    ep_rew += reward
    action_record.append(action)
print("ddpg episode reward = {}".format(ep_rew))
plt.figure(2)
plt.title('ddpg_lstm_att policy')
plt.plot(ddpg_eval_env.x_m, ddpg_eval_env.y_m, 'k^', label='target')
colors = ['red', 'green', 'blue']
for i in range(len(action_record)):
    xietong_scan_center = action_record[i]
    xietong_scan_range = [ddpg_eval_env.search_range_x, ddpg_eval_env.search_range_y]
    for j in range(ddpg_eval_env.xietong_num):
        scan_center_x = xietong_scan_center[j * 2][0] * 60.0
        scan_center_y = xietong_scan_center[j * 2 + 1][0] * 60.0
        scan_range_x = xietong_scan_range[0][j]
        scan_range_y = xietong_scan_range[1][j]
        if i == 0:
            plot_airspace(scan_range_x, scan_range_y, scan_center_x, scan_center_y, colors[j], label='Agent:{}'.format(j), linestyle='-')
        else:
            plot_airspace(scan_range_x, scan_range_y, scan_center_x, scan_center_y, colors[j], linestyle='-')
plt.legend()
plt.axis('equal')
# plt.show()

policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
n_actions = maddpg_bu0_env().action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

att_test_model = single_att_model().to(device)
# 前三个智能体的动作空间均为2，状态空间均为30
agent1 = DDPG("MlpPolicy", env=maddpg_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="D:\\result\\2023_12_18\\AS_xietong_result", verbose=1, device=device)
agent2 = DDPG("MlpPolicy", env=maddpg_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="D:\\result\\2023_12_18\\AS_xietong_result", verbose=1, device=device)
agent3 = DDPG("MlpPolicy", env=maddpg_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="D:\\result\\2023_12_18\\AS_xietong_result", verbose=1, device=device)
multi_agent = [agent1, agent2, agent3]
agent_num = np.size(multi_agent)
for i in range(agent_num):
    multi_agent[i].policy.actor.mu = deepcopy(att_test_model)
    multi_agent[i].policy.actor_target.mu = deepcopy(att_test_model)
    multi_agent[i].policy.actor_target.load_state_dict(multi_agent[i].actor.state_dict())
    multi_agent[i].policy.actor.optimizer = multi_agent[i].policy.optimizer_class(multi_agent[i].policy.actor.parameters())
    multi_policy = torch.load("./maddpg_lstm_att_fix_actor{}/policy.pth".format(i + 1))
    # multi_policy = torch.load("./maddpg_att_fix_actor{}/policy.pth".format(i + 1))
    multi_agent[i].policy.load_state_dict(multi_policy)
Agent_num = np.size(multi_agent)
maddpg_eval_env = maddpg_bu0_env()
ep_rew = 0.0
obs = maddpg_eval_env.reset()
multi_action_record = []
done = [0]
step = 0
while not int(done[0]):
    step += 1
    for i in range(Agent_num):
        action, _ = multi_agent[i].predict(obs)
        action = action.reshape(2, -1)
        multi_action_record.append(action)
        obs, reward, done, info = maddpg_eval_env.step([action[0], action[1], i])
        ep_rew += reward[0]
    ep_rew += Agent_num - 1.0
    if int(done[0]):
        # 直接判断状态空间第一个值，判断是否为零
        if abs(obs[0, 0]) > 1e-6:  # 状态空间不是全零，说明有目标未发现
            ep_rew += -20.0
        else:
            ep_rew += 20.0
print("maddpg episode reward = {}".format(ep_rew))
plt.figure(3)
plt.title('maddpg_lstm_att policy')
plt.plot(maddpg_eval_env.x_m, maddpg_eval_env.y_m, 'k^', label='target')
for i in range(len(multi_action_record)):
    xietong_scan_center = multi_action_record[i]
    xietong_scan_range = [ddpg_eval_env.search_range_x, ddpg_eval_env.search_range_y]
    scan_center_x = xietong_scan_center[0] * 60.0
    scan_center_y = xietong_scan_center[1] * 60.0
    scan_range_x = xietong_scan_range[0][i % 3]
    scan_range_y = xietong_scan_range[1][i % 3]

    if i / 3 < 1:
        plot_airspace(scan_range_x, scan_range_y, scan_center_x, scan_center_y, colors[i % 3], label='Agent:{}'.format(i % 3 + 1), linestyle='-')
    else:
        plot_airspace(scan_range_x, scan_range_y, scan_center_x, scan_center_y, colors[i % 3], linestyle='-')
plt.legend()
plt.axis('equal')
plt.show()








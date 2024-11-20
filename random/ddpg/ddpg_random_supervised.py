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
import random
from typing import Any, Dict, List, Optional, Tuple, Type, Union

log_find_nums = open("D:/decrease_model/random/ddpg/ddpg_random_supervised_findnums.csv", "w+")
log_ep_rew_mean = open("D:/decrease_model/random/ddpg/ddpg_random_supervised_ep_rew_mean.csv", "w+")
log_ep_len_mean = open("D:/decrease_model/random/ddpg/ddpg_random_supervised_ep_len_mean.csv", "w+")

Agent_actor_loss = open("D:/decrease_model/random/ddpg/Agent_actor_loss.csv", "w+")
Agent_critic_loss = open("D:/decrease_model/random/ddpg/Agent_critic_loss.csv", "w+")

dataset_num = int(1e5)
input_dataset = np.loadtxt('D:/decrease_model/random/ddpg/xietong_agent_input.csv', delimiter=',', usecols=range(60), max_rows=dataset_num)
output_dataset = np.loadtxt('D:/decrease_model/random/ddpg/xietong_agent_output.csv', delimiter=',', usecols=range(6), max_rows=dataset_num)

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

def train_agent_supervised(agent, step):
    # 直接进行有监督更新
    mse_loss = nn.MSELoss()     # 损失函数采用均方误差损失函数
    agent.policy.set_training_mode(True)

    index_start = random.randint(0, int(dataset_num / batch_size) - 1)
    input_data_sample = input_dataset[index_start * batch_size:(index_start + 1) * batch_size]
    output_data_sample = output_dataset[index_start * batch_size:(index_start + 1) * batch_size]
    input_data_tensor = torch.Tensor(input_data_sample).to(device)
    output_data_tensor = torch.Tensor(output_data_sample).to(device)
    model_output_tensor = agent.actor.mu(input_data_tensor).to(device)
    actor_loss = mse_loss(model_output_tensor, output_data_tensor)

    # 保存actor_loss
    Agent_actor_loss.write('{}\n'.format(actor_loss))
    Agent_actor_loss.flush()

    agent.actor.optimizer.zero_grad()
    actor_loss.backward(retain_graph=True)
    agent.actor.optimizer.step()

    if step % 50 == 0:
        print("Agent actor loss is {}".format(actor_loss))
    polyak_update(agent.actor.parameters(), agent.actor_target.parameters(), agent.tau)  # 更新目标网络参数
    polyak_update(agent.actor_batch_norm_stats, agent.actor_batch_norm_stats_target, 1.0)

def train_agent_critic(agent, gradient_steps, step):
    agent.policy.set_training_mode(True)
    agent._update_learning_rate([agent.actor.optimizer, agent.critic.optimizer])

    critic_loss_avg = 0.0
    for _ in range(gradient_steps):
        agent._n_updates += 1
        replay_data = agent.replay_buffer.sample(batch_size, env=agent._vec_normalize_env)

        with torch.no_grad():
            noise = replay_data.actions.clone().data.normal_(0, agent.target_policy_noise)
            noise = noise.clamp(-agent.target_noise_clip, agent.target_noise_clip)
            next_actions = (agent.actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
            next_q_values = torch.cat(agent.critic_target(replay_data.next_observations, next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * agent.gamma * next_q_values
        # Get current Q-values estimates for each critic network，使用协同评价智能体，计算协同评价网络损失critic_loss
        current_q_values = agent.critic(replay_data.observations, replay_data.actions)
        # Compute critic loss
        critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        critic_loss_avg += critic_loss
        # Optimize the critics
        agent.critic.optimizer.zero_grad()
        critic_loss.backward()
        agent.critic.optimizer.step()
        if agent._n_updates % agent.policy_delay == 0:
            polyak_update(agent.critic.parameters(), agent.critic_target.parameters(), agent.tau)
            polyak_update(agent.critic_batch_norm_stats, agent.critic_batch_norm_stats_target, 1.0)

    critic_loss_avg = critic_loss_avg / gradient_steps
    if step % 50 == 0:
        print(critic_loss_avg)
    # critic_loss_avg
    Agent_critic_loss.write('{}\n'.format(critic_loss_avg))
    Agent_critic_loss.flush()

def evaluate_policy(agent):
    single_agent_collect_env = xietong_bu0_env()
    reward = 0
    obs = single_agent_collect_env.reset()
    dones = 0
    steps = 0
    while not int(dones):
        steps += 1
        actions, _ = agent.predict(obs)
        actions = actions.reshape(6, -1)
        obs, rewards, dones, infos = single_agent_collect_env.step(actions)
        reward += rewards
        if dones:  # 到达最大步数且未完成
            log_ep_len_mean.write('{}\n'.format(steps))
            log_ep_len_mean.flush()
            log_ep_rew_mean.write('{}\n'.format(reward))
            log_ep_rew_mean.flush()
            return reward

# 创建一个SaveCallback，用于在训练过程中保存模型
def save_model_callback(_locals, _globals):
    # 获取当前训练步数
    current_step = _locals["self"].num_timesteps
    save_interval = 10000  # 每10000步保存一次模型
    if current_step % save_interval == 0:
        # 设置模型保存路径和文件名
        model_save_path = f"D:/decrease_model/saved_model/ddpg/ddpg_random/ddpg_random_{current_step:.1e}.zip"
        # 保存模型
        _locals["self"].save(model_save_path)
        # print(f"Model saved at step {current_step}.")

# 超参数
gradient_steps = 10
batch_size = 100

total_timesteps = int(1e6)
callback = None
log_interval = 50
eval_env = None
eval_freq = -1
n_eval_episodes = 5
tb_log_name = 'DDPG'
eval_log_path = None
reset_num_timesteps = True
progress_bar = False
train_freq: Union[int, Tuple[int, str]] = (1, "step")


multi_total_timesteps = []
multi_callback = []
multi_rollout = []

# 大循环
reward_avg = 0.0
step = 0

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
n_actions = xietong_bu0_env().action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

agent_pre_train = DDPG("MlpPolicy", env=xietong_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, verbose=1, device=device)

agent_pre_train.train_freq = train_freq
# total_timesteps, callback = agent_pre_train._setup_learn(total_timesteps, eval_env, callback, eval_freq,
#                                                     n_eval_episodes, eval_log_path, reset_num_timesteps, tb_log_name)
total_timesteps, callback = agent_pre_train._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar,)
callback.on_training_start(locals(), globals())
agent_pre_train._setup_model()

while step < 3e4:

    step += 1

    if step < 1e4:

        train_agent_supervised(agent_pre_train, step)

    else:

        rollout = agent_pre_train.collect_rollouts(
                agent_pre_train.env,
                train_freq=agent_pre_train.train_freq,
                action_noise=agent_pre_train.action_noise,
                callback=callback,
                learning_starts=agent_pre_train.learning_starts,
                replay_buffer=agent_pre_train.replay_buffer,
                log_interval=log_interval,
            )

        if step < 3e4:    # 首先只更新评价网络
            if agent_pre_train.num_timesteps > 0 and agent_pre_train.num_timesteps > agent_pre_train.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = agent_pre_train.gradient_steps if agent_pre_train.gradient_steps >= 0 else rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    train_agent_critic(agent_pre_train, gradient_steps, step)

    if step % 5 == 0:
        reward_avg += evaluate_policy(agent_pre_train)
    if step % 100 == 0:
        print(reward_avg / 20, step, agent_pre_train.num_timesteps)
        reward_avg = 0.0

agent_supervised = DDPG("MlpPolicy", env=xietong_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs, learning_rate=0.0001,
            learning_starts=500, tensorboard_log="D:/decrease_model/result/ddpg_random_supervised", verbose=1, device=device)

agent_supervised.policy.actor.mu = deepcopy(agent_pre_train.policy.actor.mu)
agent_supervised.policy.actor_target.mu = deepcopy(agent_pre_train.policy.actor_target.mu)
agent_supervised.policy.actor_target.load_state_dict(agent_supervised.policy.actor.state_dict())
agent_supervised.policy.actor.optimizer = deepcopy(agent_pre_train.policy.actor.optimizer)

agent_supervised.learn(total_timesteps=int(8e5), callback=save_model_callback, log_interval=50)
agent_supervised.save("ddpg_random.zip")




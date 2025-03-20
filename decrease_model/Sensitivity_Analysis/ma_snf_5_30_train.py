import gym
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from stable_baselines3 import DDPG
import torch
from stable_baselines3.common.noise import NormalActionNoise, VectorizedActionNoise
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.type_aliases import RolloutReturn
from stable_baselines3.common.utils import polyak_update
from torch.nn import functional as F
from gym import spaces
import torch.nn as nn
import random

action_dim = 2
log_find_nums = open("./maddpg/ma_snf_5_30_nums.csv", "w+")
log_ep_rew_mean = open("./maddpg/ma_snf_5_30_rew.csv", "w+")
log_ep_len_mean = open("./maddpg/ma_snf_5_30_len.csv", "w+")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

dataset_num = int(1e5)
input_data = np.loadtxt('log_Agent2_input.csv', delimiter=',', usecols=range(60), max_rows=dataset_num)
output_data = np.loadtxt('log_Agent2_output.csv', delimiter=',', usecols=range(2), max_rows=dataset_num)
input_dataset = [input_data, input_data, input_data, input_data, input_data]
output_dataset = [output_data, output_data, output_data, output_data, output_data]

class single_bu0_env(gym.Env):
    """A kongyu huafen environment for OpenAI gym"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(single_bu0_env, self).__init__()

        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(2,), dtype=np.float16)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(60,), dtype=np.float16)

        self.target_num = 30  # 目标数目
        self.xietong_num = 1  # 协同数目
        self.search_range_x = [20, 20, 20, 20, 20]  # 方位搜索范围
        self.search_range_y = [20, 20, 20, 20, 20]  # 俯仰搜索范围

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

        self._max_episode_steps = 50  # 单个最大搜索是30次
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
        reward5 = -1.0
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
        self.xietong_num = 1  # 协同数目
        self.search_range_x = [20, 20, 20, 20, 20]  # 方位搜索范围
        self.search_range_y = [20, 20, 20, 20, 20]  # 俯仰搜索范围

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

        self._max_episode_steps = 50  # 单个最大搜索是30次
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
        reward5 = -1.0
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

        self.action_space = spaces.Box(low=-1.0, high=+1.0, shape=(10,), dtype=np.float16)
        self.observation_space = spaces.Box(low=-100, high=100, shape=(60,), dtype=np.float16)

    def reset(self):
        return np.zeros((1, 60))

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

def train_agent(multi_agent, xietong_agent, gradient_steps):
    Agent_num = np.size(multi_agent)

    xietong_agent.policy.set_training_mode(True)
    xietong_agent._update_learning_rate([xietong_agent.actor.optimizer, xietong_agent.critic.optimizer])
    for i in range(Agent_num):
        multi_agent[i].policy.set_training_mode(True)
        multi_agent[i]._update_learning_rate([multi_agent[i].actor.optimizer, multi_agent[i].critic.optimizer])

    for _ in range(gradient_steps):
        xietong_agent._n_updates += 1
        # Sample replay buffer，从协同智能体里采轨迹
        replay_data = xietong_agent.replay_buffer.sample(batch_size, env=xietong_agent._vec_normalize_env)

        multi_next_actions = []
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            for i in range(Agent_num):
                noise = replay_data.actions[:, 0:2].clone().data.normal_(0, multi_agent[i].target_policy_noise)
                noise = noise.clamp(-multi_agent[i].target_noise_clip, multi_agent[i].target_noise_clip)
                next_actions = (multi_agent[i].actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                multi_next_actions.append(deepcopy(next_actions))
            multi_next_actions = torch.cat(multi_next_actions, dim=1)
            # Compute the next Q-values: min over all critics targets
            next_q_values = torch.cat(xietong_agent.critic_target(replay_data.next_observations, multi_next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * xietong_agent.gamma * next_q_values

        # Get current Q-values estimates for each critic network，使用协同评价智能体，计算协同评价网络损失critic_loss
        current_q_values = xietong_agent.critic(replay_data.observations, replay_data.actions)
        # Compute critic loss
        critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        # Optimize the critics
        xietong_agent.critic.optimizer.zero_grad()
        critic_loss.backward()
        xietong_agent.critic.optimizer.step()

        # 更新各动作智能体动作网络
        multi_actor_loss = []
        for i in range(Agent_num):
            multi_actor_loss.append([])
        if xietong_agent._n_updates % xietong_agent.policy_delay == 0:
            for i in range(Agent_num):
                # 重新选择对应智能体动作
                action_rechoose = multi_agent[i].actor(replay_data.observations)
                replay_data.actions[:, 2 * i : 2 * (i + 1)] = action_rechoose
                multi_actor_loss[i] = -xietong_agent.critic.q1_forward(replay_data.observations, replay_data.actions).mean()

                # Optimize the actor
                multi_agent[i].actor.optimizer.zero_grad()
                multi_actor_loss[i].backward(retain_graph=True)
                multi_agent[i].actor.optimizer.step()

                polyak_update(multi_agent[i].actor.parameters(), multi_agent[i].actor_target.parameters(), multi_agent[i].tau)
                polyak_update(multi_agent[i].actor_batch_norm_stats, multi_agent[i].actor_batch_norm_stats_target, 1.0)
            polyak_update(xietong_agent.critic.parameters(), xietong_agent.critic_target.parameters(), xietong_agent.tau)
            polyak_update(xietong_agent.critic_batch_norm_stats, xietong_agent.critic_batch_norm_stats_target, 1.0)

def train_agent_supervised(multi_agent, step):
    # 直接进行有监督更新
    Agent_num = np.size(multi_agent)
    mse_loss = nn.MSELoss()     # 损失函数采用均方误差损失函数
    multi_actor_loss = []
    for i in range(Agent_num):
        multi_actor_loss.append([])
    for i in range(Agent_num):
        index_start = random.randint(0, int(dataset_num / batch_size) - 1)
        input_data_sample = input_dataset[i][index_start * batch_size:(index_start + 1) * batch_size]
        output_data_sample = output_dataset[i][index_start * batch_size:(index_start + 1) * batch_size]
        input_data_tensor = torch.Tensor(input_data_sample).to(device)
        output_data_tensor = torch.Tensor(output_data_sample).to(device)
        model_output_tensor = multi_agent[i].actor.mu(input_data_tensor).to(device)
        multi_actor_loss[i] = mse_loss(model_output_tensor, output_data_tensor)

        # # 保存actor_loss
        # Agent_actor_loss[i].write('{}\n'.format(multi_actor_loss[i]))
        # Agent_actor_loss[i].flush()

        multi_agent[i].actor.optimizer.zero_grad()
        multi_actor_loss[i].backward(retain_graph=True)
        multi_agent[i].actor.optimizer.step()

        if step % 50 == 0:
            print("Agent {} actor loss is {}".format(i, multi_actor_loss[i]))
        polyak_update(multi_agent[i].actor.parameters(), multi_agent[i].actor_target.parameters(), multi_agent[i].tau)  # 更新目标网络参数

def train_xietong_agent_critic(multi_agent, xietong_agent, gradient_steps, step):   # 只更新评价网络
    Agent_num = np.size(multi_agent)

    xietong_agent.policy.set_training_mode(True)
    xietong_agent._update_learning_rate([xietong_agent.actor.optimizer, xietong_agent.critic.optimizer])
    for i in range(Agent_num):
        multi_agent[i].policy.set_training_mode(True)
        multi_agent[i]._update_learning_rate([multi_agent[i].actor.optimizer, multi_agent[i].critic.optimizer])

    critic_loss_avg = 0.0
    for _ in range(gradient_steps):
        xietong_agent._n_updates += 1
        # Sample replay buffer，从协同智能体里采轨迹
        replay_data = xietong_agent.replay_buffer.sample(batch_size, env=xietong_agent._vec_normalize_env)

        multi_next_actions = []
        with torch.no_grad():
            # Select action according to policy and add clipped noise
            for i in range(Agent_num):
                noise = replay_data.actions[:, 0:2].clone().data.normal_(0, multi_agent[i].target_policy_noise)
                noise = noise.clamp(-multi_agent[i].target_noise_clip, multi_agent[i].target_noise_clip)
                next_actions = (multi_agent[i].actor_target(replay_data.next_observations) + noise).clamp(-1, 1)
                multi_next_actions.append(deepcopy(next_actions))
            multi_next_actions = torch.cat(multi_next_actions, dim=1)
            # Compute the next Q-values: min over all critics targets
            next_q_values = torch.cat(xietong_agent.critic_target(replay_data.next_observations, multi_next_actions), dim=1)
            next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
            target_q_values = replay_data.rewards + (1 - replay_data.dones) * xietong_agent.gamma * next_q_values

        # Get current Q-values estimates for each critic network，使用协同评价智能体，计算协同评价网络损失critic_loss
        current_q_values = xietong_agent.critic(replay_data.observations, replay_data.actions)
        # Compute critic loss
        critic_loss = sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
        critic_loss_avg += critic_loss
        # Optimize the critics
        xietong_agent.critic.optimizer.zero_grad()
        critic_loss.backward()
        xietong_agent.critic.optimizer.step()

        if xietong_agent._n_updates % xietong_agent.policy_delay == 0:
            polyak_update(xietong_agent.critic.parameters(), xietong_agent.critic_target.parameters(), xietong_agent.tau)

    critic_loss_avg = critic_loss_avg / gradient_steps
    if step % 50 == 0:
        print(critic_loss_avg)
    # critic_loss_avg
    # Agent_critic_loss.write('{}\n'.format(critic_loss_avg))
    # Agent_critic_loss.flush()

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


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
policy_kwargs = dict(activation_fn=torch.nn.ReLU, net_arch=[128, 64])
n_actions = single_bu0_env().action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 前三个智能体的动作空间均为2，状态空间均为30
agent1 = DDPG("MlpPolicy", env=single_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs,
              learning_rate=0.0001, learning_starts=500, verbose=1, device=device)
agent2 = DDPG("MlpPolicy", env=single_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs,
              learning_rate=0.0001, learning_starts=500, verbose=1, device=device)
agent3 = DDPG("MlpPolicy", env=single_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs,
              learning_rate=0.0001, learning_starts=500, verbose=1, device=device)
agent4 = DDPG("MlpPolicy", env=single_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs,
              learning_rate=0.0001, learning_starts=500, verbose=1, device=device)
agent5 = DDPG("MlpPolicy", env=single_bu0_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs,
              learning_rate=0.0001, learning_starts=500, verbose=1, device=device)
multi_agent = [agent1, agent2, agent3, agent4, agent5]
agent_num = np.size(multi_agent)
# 协同智能体动作空间为6，状态空间为30
n_actions = xietong_env().action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
xietong_agent = DDPG("MlpPolicy", env=xietong_env(), gamma=0.98, action_noise=action_noise, policy_kwargs=policy_kwargs,
                     learning_rate=0.0001, learning_starts=500, verbose=1, device=device)  # 单个智能体只用critic网络

# 超参数
gradient_steps = 10
batch_size = 100

total_timesteps = int(5e6)
callback = None
log_interval = 4
eval_env = None
eval_freq = -1
n_eval_episodes = 5
tb_log_name = 'DDPG'
eval_log_path = None
reset_num_timesteps = True
progress_bar = False

multi_total_timesteps = []
multi_callback = []
multi_rollout = []

# 大循环
reward_avg = 0.0
step = int(0.0)

for i in range(agent_num):
    total_timesteps, callback = multi_agent[i]._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar,)
    callback.on_training_start(locals(), globals())
    multi_total_timesteps.append(deepcopy(total_timesteps))
    # multi_callback.append(deepcopy(callback))
    multi_callback.append(callback)

xietong_total_timesteps, xietong_callback = xietong_agent._setup_learn(total_timesteps, callback, reset_num_timesteps, tb_log_name, progress_bar,)
xietong_callback.on_training_start(locals(), globals())

while xietong_agent.num_timesteps < total_timesteps:   # 三个智能体设置的训练步长是一样的

    step += 1

    if step < 5e3:

        train_agent_supervised(multi_agent, step)

    else:

        xietong_rollout = collect_xietong_rollouts(multi_agent, xietong_agent, multi_callback, xietong_callback)

        if step < 2.5e4:  # 首先只更新评价网络
            if xietong_agent.num_timesteps > 0 and xietong_agent.num_timesteps > xietong_agent.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = xietong_agent.gradient_steps if xietong_agent.gradient_steps >= 0 else xietong_rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    train_xietong_agent_critic(multi_agent, xietong_agent, gradient_steps, step)

        else:
            if xietong_agent.num_timesteps > 0 and xietong_agent.num_timesteps > xietong_agent.learning_starts:
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = xietong_agent.gradient_steps if xietong_agent.gradient_steps >= 0 else xietong_rollout.episode_timesteps
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    train_agent(multi_agent, xietong_agent, gradient_steps)

    # if step % 1000 == 0:
    #     multi_agent[0].save(f".\maddpg\save_model\ma_snf_actor1_{step:.1e}.zip")
    #     multi_agent[1].save(f".\maddpg\save_model\ma_snf_actor2_{step:.1e}.zip")
    #     multi_agent[2].save(f".\maddpg\save_model\ma_snf_actor3_{step:.1e}.zip")
    #     # multi_agent[3].save(f".\maddpg\save_model\ma_snf_actor4_{step:.1e}.zip")
    #     xietong_agent.save(f".\maddpg\save_model\ma_snf_critic_{step:.1e}.zip")

    if step % 5 == 0:
        reward_avg += evaluate_policy(multi_agent)
    if step % 100 == 0:
        print(reward_avg / 20, step, xietong_agent.num_timesteps)
        reward_avg = 0.0

multi_agent[0].save("ma_snf_actor1")
multi_agent[1].save("ma_snf_actor2")
multi_agent[2].save("ma_snf_actor3")
xietong_agent.save("ma_snf_critic")

# agent1.load("maddpg_fix_actor1.zip")
# agent2.load("maddpg_fix_actor2.zip")
# agent3.load("maddpg_fix_actor3.zip")
# xietong_agent.load("maddpg_fix_critic.zip")
# multi_agent = [agent1, agent2, agent3]
#
# eval_rew = evaluate_policy(multi_agent)
# print(eval_rew)
































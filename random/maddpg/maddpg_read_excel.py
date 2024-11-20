import csv
import matplotlib.pyplot as plt
import numpy as np

maddpg_random_findnums_data = np.loadtxt('maddpg_random_findnums.csv', delimiter=',')
maddpg_random_supervised_findnums_data = np.loadtxt('maddpg_random_supervised_findnums.csv', delimiter=',')
maddpg_att_random_supervised_findnums_data = np.loadtxt('maddpg_att_random_supervised_findnums.csv', delimiter=',')
maddpg_lstm_att_random_supervised_findnums_data = np.loadtxt('maddpg_lstm_att_random_supervised_findnums.csv', delimiter=',')
avg_steps = 50
maddpg_random_findnums = np.zeros((int(np.size(maddpg_random_findnums_data) / avg_steps)))
maddpg_random_supervised_findnums = np.zeros((int(np.size(maddpg_random_supervised_findnums_data) / avg_steps),))
maddpg_att_random_supervised_findnums = np.zeros((int(np.size(maddpg_att_random_supervised_findnums_data) / avg_steps),))
maddpg_lstm_att_random_supervised_findnums = np.zeros((int(np.size(maddpg_lstm_att_random_supervised_findnums_data) / avg_steps),))
for i in range(np.size(maddpg_random_findnums)):
    for j in range(avg_steps):
        maddpg_random_findnums[i] += maddpg_random_findnums_data[i * avg_steps + j]
    maddpg_random_findnums[i] = maddpg_random_findnums[i] / avg_steps
for i in range(np.size(maddpg_random_supervised_findnums)):
    for j in range(avg_steps):
        maddpg_random_supervised_findnums[i] += maddpg_random_supervised_findnums_data[i * avg_steps + j]
    maddpg_random_supervised_findnums[i] = maddpg_random_supervised_findnums[i] / avg_steps
for i in range(np.size(maddpg_att_random_supervised_findnums)):
    for j in range(avg_steps):
        maddpg_att_random_supervised_findnums[i] += maddpg_att_random_supervised_findnums_data[i * avg_steps + j]
    maddpg_att_random_supervised_findnums[i] = maddpg_att_random_supervised_findnums[i] / avg_steps
for i in range(np.size(maddpg_lstm_att_random_supervised_findnums)):
    for j in range(avg_steps):
        maddpg_lstm_att_random_supervised_findnums[i] += maddpg_lstm_att_random_supervised_findnums_data[i * avg_steps + j]
    maddpg_lstm_att_random_supervised_findnums[i] = maddpg_lstm_att_random_supervised_findnums[i] / avg_steps
x_label1 = np.arange(0, np.size(maddpg_random_findnums), 1)
x_label2 = np.arange(0, np.size(maddpg_random_supervised_findnums), 1)
x_label3 = np.arange(0, np.size(maddpg_att_random_supervised_findnums), 1)
x_label4 = np.arange(0, np.size(maddpg_lstm_att_random_supervised_findnums), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), maddpg_random_findnums, 'k', label='maddpg')
plt.plot(x_label2 * 1e6 / np.size(x_label2), maddpg_random_supervised_findnums, 'g', label='maddpg_supervised')
plt.plot(x_label3 * 1e6 / np.size(x_label3), maddpg_att_random_supervised_findnums, 'b', label='maddpg_att_supervised')
plt.plot(x_label4 * 1e6 / np.size(x_label4), maddpg_lstm_att_random_supervised_findnums, 'r', label='maddpg_lstm_att_supervised')
plt.xlabel("time_steps")
plt.ylabel("episode_find_nums")
plt.legend()
# plt.show()

maddpg_random_ep_rew_mean_data = np.loadtxt('maddpg_random_ep_rew_mean.csv', delimiter=',')
maddpg_random_supervised_ep_rew_mean_data = np.loadtxt('maddpg_random_supervised_ep_rew_mean.csv', delimiter=',')
maddpg_att_random_supervised_ep_rew_mean_data = np.loadtxt('maddpg_att_random_supervised_ep_rew_mean.csv', delimiter=',')
maddpg_lstm_att_random_supervised_ep_rew_mean_data = np.loadtxt('maddpg_lstm_att_random_supervised_ep_rew_mean.csv', delimiter=',')
avg_steps = 50
maddpg_random_ep_rew_mean = np.zeros((int(np.size(maddpg_random_ep_rew_mean_data) / avg_steps),))
maddpg_random_supervised_ep_rew_mean = np.zeros((int(np.size(maddpg_random_supervised_ep_rew_mean_data) / avg_steps),))
maddpg_att_random_supervised_ep_rew_mean = np.zeros((int(np.size(maddpg_att_random_supervised_ep_rew_mean_data) / avg_steps),))
maddpg_lstm_att_random_supervised_ep_rew_mean = np.zeros((int(np.size(maddpg_lstm_att_random_supervised_ep_rew_mean_data) / avg_steps),))
for i in range(np.size(maddpg_random_ep_rew_mean)):
    for j in range(avg_steps):
        maddpg_random_ep_rew_mean[i] += maddpg_random_ep_rew_mean_data[i * avg_steps + j]
    maddpg_random_ep_rew_mean[i] = maddpg_random_ep_rew_mean[i] / avg_steps
for i in range(np.size(maddpg_random_supervised_ep_rew_mean)):
    for j in range(avg_steps):
        maddpg_random_supervised_ep_rew_mean[i] += maddpg_random_supervised_ep_rew_mean_data[i * avg_steps + j]
    maddpg_random_supervised_ep_rew_mean[i] = maddpg_random_supervised_ep_rew_mean[i] / avg_steps
for i in range(np.size(maddpg_att_random_supervised_ep_rew_mean)):
    for j in range(avg_steps):
        maddpg_att_random_supervised_ep_rew_mean[i] += maddpg_att_random_supervised_ep_rew_mean_data[i * avg_steps + j]
    maddpg_att_random_supervised_ep_rew_mean[i] = maddpg_att_random_supervised_ep_rew_mean[i] / avg_steps
for i in range(np.size(maddpg_lstm_att_random_supervised_ep_rew_mean)):
    for j in range(avg_steps):
        maddpg_lstm_att_random_supervised_ep_rew_mean[i] += maddpg_lstm_att_random_supervised_ep_rew_mean_data[i * avg_steps + j]
    maddpg_lstm_att_random_supervised_ep_rew_mean[i] = maddpg_lstm_att_random_supervised_ep_rew_mean[i] / avg_steps
x_label1 = np.arange(0, np.size(maddpg_random_ep_rew_mean), 1)
x_label2 = np.arange(0, np.size(maddpg_random_supervised_ep_rew_mean), 1)
x_label3 = np.arange(0, np.size(maddpg_att_random_supervised_ep_rew_mean), 1)
x_label4 = np.arange(0, np.size(maddpg_lstm_att_random_supervised_ep_rew_mean), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), maddpg_random_ep_rew_mean, 'k', label='maddpg')
plt.plot(x_label2 * 1e6 / np.size(x_label2), maddpg_random_supervised_ep_rew_mean, 'g', label='maddpg_supervised')
plt.plot(x_label3 * 1e6 / np.size(x_label3), maddpg_att_random_supervised_ep_rew_mean, 'b', label='maddpg_att_supervised')
plt.plot(x_label4 * 1e6 / np.size(x_label4), maddpg_lstm_att_random_supervised_ep_rew_mean, 'r', label='maddpg_lstm_att_supervised')
plt.xlabel("time_steps")
plt.ylabel("ep_rew_mean")
plt.legend()
# plt.show()

maddpg_random_ep_len_mean_data = np.loadtxt('maddpg_random_ep_len_mean.csv', delimiter=',')
maddpg_random_supervised_ep_len_mean_data = np.loadtxt('maddpg_random_supervised_ep_len_mean.csv', delimiter=',')
maddpg_att_random_supervised_ep_len_mean_data = np.loadtxt('maddpg_att_random_supervised_ep_len_mean.csv', delimiter=',')
maddpg_lstm_att_random_supervised_ep_len_mean_data = np.loadtxt('maddpg_lstm_att_random_supervised_ep_len_mean.csv', delimiter=',')
avg_steps = 50
maddpg_random_ep_len_mean = np.zeros((int(np.size(maddpg_random_ep_len_mean_data) / avg_steps),))
maddpg_random_supervised_ep_len_mean = np.zeros((int(np.size(maddpg_random_supervised_ep_len_mean_data) / avg_steps),))
maddpg_att_random_supervised_ep_len_mean = np.zeros((int(np.size(maddpg_att_random_supervised_ep_len_mean_data) / avg_steps),))
maddpg_lstm_att_random_supervised_ep_len_mean = np.zeros((int(np.size(maddpg_lstm_att_random_supervised_ep_len_mean_data) / avg_steps),))
for i in range(np.size(maddpg_random_ep_len_mean)):
    for j in range(avg_steps):
        maddpg_random_ep_len_mean[i] += maddpg_random_ep_len_mean_data[i * avg_steps + j]
    maddpg_random_ep_len_mean[i] = maddpg_random_ep_len_mean[i] / avg_steps
for i in range(np.size(maddpg_random_supervised_ep_len_mean)):
    for j in range(avg_steps):
        maddpg_random_supervised_ep_len_mean[i] += maddpg_random_supervised_ep_len_mean_data[i * avg_steps + j]
    maddpg_random_supervised_ep_len_mean[i] = maddpg_random_supervised_ep_len_mean[i] / avg_steps
for i in range(np.size(maddpg_att_random_supervised_ep_len_mean)):
    for j in range(avg_steps):
        maddpg_att_random_supervised_ep_len_mean[i] += maddpg_att_random_supervised_ep_len_mean_data[i * avg_steps + j]
    maddpg_att_random_supervised_ep_len_mean[i] = maddpg_att_random_supervised_ep_len_mean[i] / avg_steps
for i in range(np.size(maddpg_lstm_att_random_supervised_ep_len_mean)):
    for j in range(avg_steps):
        maddpg_lstm_att_random_supervised_ep_len_mean[i] += maddpg_lstm_att_random_supervised_ep_len_mean_data[i * avg_steps + j]
    maddpg_lstm_att_random_supervised_ep_len_mean[i] = maddpg_lstm_att_random_supervised_ep_len_mean[i] / avg_steps
z_label1 = np.arange(0, np.size(maddpg_random_ep_len_mean), 1)
x_label2 = np.arange(0, np.size(maddpg_random_supervised_ep_len_mean), 1)
x_label3 = np.arange(0, np.size(maddpg_att_random_supervised_ep_len_mean), 1)
x_label4 = np.arange(0, np.size(maddpg_lstm_att_random_supervised_ep_len_mean), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), maddpg_random_ep_len_mean, 'k', label='maddpg')
plt.plot(x_label2 * 1e6 / np.size(x_label2), maddpg_random_supervised_ep_len_mean, 'g', label='maddpg_supervised')
plt.plot(x_label3 * 1e6 / np.size(x_label3), maddpg_att_random_supervised_ep_len_mean, 'b', label='maddpg_att_supervised')
plt.plot(x_label4 * 1e6 / np.size(x_label4), maddpg_lstm_att_random_supervised_ep_len_mean, 'r', label='maddpg_lstm_att_supervised')
plt.xlabel("time_steps")
plt.ylabel("ep_len_mean")
plt.legend()
# plt.show()

Agent1_actor_loss_data = np.loadtxt('Agent1_actor_loss.csv', delimiter=',')
Agent2_actor_loss_data = np.loadtxt('Agent2_actor_loss.csv', delimiter=',')
Agent3_actor_loss_data = np.loadtxt('Agent3_actor_loss.csv', delimiter=',')
avg_steps = 10
Agent1_actor_loss = np.zeros((int(np.size(Agent1_actor_loss_data) / avg_steps),))
Agent2_actor_loss = np.zeros((int(np.size(Agent2_actor_loss_data) / avg_steps),))
Agent3_actor_loss = np.zeros((int(np.size(Agent3_actor_loss_data) / avg_steps),))
for i in range(np.size(Agent1_actor_loss)):
    for j in range(avg_steps):
        Agent1_actor_loss[i] += Agent1_actor_loss_data[i * avg_steps + j]
        Agent2_actor_loss[i] += Agent2_actor_loss_data[i * avg_steps + j]
        Agent3_actor_loss[i] += Agent3_actor_loss_data[i * avg_steps + j]
    Agent1_actor_loss[i] = Agent1_actor_loss[i] / avg_steps
    Agent2_actor_loss[i] = Agent2_actor_loss[i] / avg_steps
    Agent3_actor_loss[i] = Agent3_actor_loss[i] / avg_steps
x_label1 = np.arange(0, np.size(Agent1_actor_loss), 1)
x_label2 = np.arange(0, np.size(Agent2_actor_loss), 1)
x_label3 = np.arange(0, np.size(Agent3_actor_loss), 1)
plt.figure()
plt.plot(x_label1 * 1e5 / np.size(x_label1), Agent1_actor_loss, 'b', label='NN_1_loss')
plt.plot(x_label2 * 1e5 / np.size(x_label2), Agent2_actor_loss, 'g', label='NN_2_loss')
plt.plot(x_label3 * 1e5 / np.size(x_label3), Agent3_actor_loss, 'r', label='NN_3_loss')
plt.xlabel("time_steps")
plt.ylabel("NN_loss")
# plt.title('maddpg_att_random_supervised')
plt.legend()
# plt.show()

Agent_critic_loss_data = np.loadtxt('Agent_critic_loss.csv', delimiter=',')
avg_steps = 30
Agent_critic_loss = np.zeros((int(np.size(Agent_critic_loss_data) / avg_steps),))
for i in range(np.size(Agent_critic_loss)):
    for j in range(avg_steps):
        Agent_critic_loss[i] += Agent_critic_loss_data[i * avg_steps + j]
    Agent_critic_loss[i] = Agent_critic_loss[i] / avg_steps
x_label1 = np.arange(0, np.size(Agent_critic_loss), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), Agent_critic_loss, 'b', label='Agent_critic_loss')
plt.xlabel("time_steps")
plt.ylabel("critic_loss")
plt.title('maddpg_att_random_supervised')
plt.legend()
# plt.show()

Agent1_actor_loss_data = np.loadtxt('Agent1_lstm_actor_loss.csv', delimiter=',')
Agent2_actor_loss_data = np.loadtxt('Agent2_lstm_actor_loss.csv', delimiter=',')
Agent3_actor_loss_data = np.loadtxt('Agent3_lstm_actor_loss.csv', delimiter=',')
avg_steps = 10
Agent1_actor_loss = np.zeros((int(np.size(Agent1_actor_loss_data) / avg_steps),))
Agent2_actor_loss = np.zeros((int(np.size(Agent2_actor_loss_data) / avg_steps),))
Agent3_actor_loss = np.zeros((int(np.size(Agent3_actor_loss_data) / avg_steps),))
for i in range(np.size(Agent1_actor_loss)):
    for j in range(avg_steps):
        Agent1_actor_loss[i] += Agent1_actor_loss_data[i * avg_steps + j]
        Agent2_actor_loss[i] += Agent2_actor_loss_data[i * avg_steps + j]
        Agent3_actor_loss[i] += Agent3_actor_loss_data[i * avg_steps + j]
    Agent1_actor_loss[i] = Agent1_actor_loss[i] / avg_steps
    Agent2_actor_loss[i] = Agent2_actor_loss[i] / avg_steps
    Agent3_actor_loss[i] = Agent3_actor_loss[i] / avg_steps
x_label1 = np.arange(0, np.size(Agent1_actor_loss), 1)
x_label2 = np.arange(0, np.size(Agent2_actor_loss), 1)
x_label3 = np.arange(0, np.size(Agent3_actor_loss), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), Agent1_actor_loss, 'b', label='Agent1_lstm_actor_loss')
plt.plot(x_label2 * 1e6 / np.size(x_label2), Agent2_actor_loss, 'g', label='Agent2_lstm_actor_loss')
plt.plot(x_label3 * 1e6 / np.size(x_label3), Agent3_actor_loss, 'r', label='Agent3_lstm_actor_loss')
plt.xlabel("time_steps")
plt.ylabel("actor_loss")
plt.title('maddpg_lstm_att_random_supervised')
plt.legend()
# plt.show()

Agent_critic_loss_data = np.loadtxt('Agent_lstm_critic_loss.csv', delimiter=',')
avg_steps = 30
Agent_critic_loss = np.zeros((int(np.size(Agent_critic_loss_data) / avg_steps),))
for i in range(np.size(Agent_critic_loss)):
    for j in range(avg_steps):
        Agent_critic_loss[i] += Agent_critic_loss_data[i * avg_steps + j]
    Agent_critic_loss[i] = Agent_critic_loss[i] / avg_steps
x_label1 = np.arange(0, np.size(Agent_critic_loss), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), Agent_critic_loss, 'b', label='Agent_lstm_critic_loss')
plt.xlabel("time_steps")
plt.ylabel("critic_loss")
plt.title('maddpg_lstm_att_random_supervised')
plt.legend()

last_steps = int(1e3)
maddpg_random_supervised_ep_findnums_avg = sum(maddpg_random_supervised_findnums_data[np.size(maddpg_random_supervised_findnums_data) - last_steps:np.size(maddpg_random_supervised_findnums_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(maddpg_random_supervised_findnums_data) - last_steps, np.size(maddpg_random_supervised_findnums_data)):
    sum1 += (maddpg_random_supervised_findnums_data[i] - maddpg_random_supervised_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
last_steps = int(1000)
maddpg_random_supervised_ep_rew_avg = sum(maddpg_random_supervised_ep_rew_mean_data[np.size(maddpg_random_supervised_ep_rew_mean_data) - last_steps:np.size(maddpg_random_supervised_ep_rew_mean_data)]) / last_steps
sum2 = 0.0
for i in range(np.size(maddpg_random_supervised_ep_rew_mean_data) - last_steps, np.size(maddpg_random_supervised_ep_rew_mean_data)):
    sum2 += (maddpg_random_supervised_ep_rew_mean_data[i] - maddpg_random_supervised_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
maddpg_random_supervised_ep_len_avg = sum(maddpg_random_supervised_ep_len_mean_data[np.size(maddpg_random_supervised_ep_len_mean_data) - last_steps:np.size(maddpg_random_supervised_ep_len_mean_data)]) / last_steps
sum3 = 0.0
for i in range(np.size(maddpg_random_supervised_ep_len_mean_data) - last_steps, np.size(maddpg_random_supervised_ep_len_mean_data)):
    sum3 += (maddpg_random_supervised_ep_len_mean_data[i] - maddpg_random_supervised_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('maddpg_random_supervised_ep_findnums_avg: {}, sigma: {}'.format(maddpg_random_supervised_ep_findnums_avg, sigma1))
print('maddpg_random_supervised_ep_rew_avg: {}, sigma: {}'.format(maddpg_random_supervised_ep_rew_avg, sigma2))
print('maddpg_random_supervised_ep_len_avg: {}, sigma: {}'.format(maddpg_random_supervised_ep_len_avg, sigma3))

last_steps = int(1e3)
maddpg_att_random_supervised_ep_findnums_avg = sum(maddpg_att_random_supervised_findnums_data[np.size(maddpg_att_random_supervised_findnums_data) - last_steps:np.size(maddpg_att_random_supervised_findnums_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(maddpg_att_random_supervised_findnums_data) - last_steps, np.size(maddpg_att_random_supervised_findnums_data)):
    sum1 += (maddpg_att_random_supervised_findnums_data[i] - maddpg_att_random_supervised_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
last_steps = int(1000)
maddpg_att_random_supervised_ep_rew_avg = sum(maddpg_att_random_supervised_ep_rew_mean_data[np.size(maddpg_att_random_supervised_ep_rew_mean_data) - last_steps:np.size(maddpg_att_random_supervised_ep_rew_mean_data)]) / last_steps
sum2 = 0.0
for i in range(np.size(maddpg_att_random_supervised_ep_rew_mean_data) - last_steps, np.size(maddpg_att_random_supervised_ep_rew_mean_data)):
    sum2 += (maddpg_att_random_supervised_ep_rew_mean_data[i] - maddpg_att_random_supervised_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
maddpg_att_random_supervised_ep_len_avg = sum(maddpg_att_random_supervised_ep_len_mean_data[np.size(maddpg_att_random_supervised_ep_len_mean_data) - last_steps:np.size(maddpg_att_random_supervised_ep_len_mean_data)]) / last_steps
sum3 = 0.0
for i in range(np.size(maddpg_att_random_supervised_ep_len_mean_data) - last_steps, np.size(maddpg_att_random_supervised_ep_len_mean_data)):
    sum3 += (maddpg_att_random_supervised_ep_len_mean_data[i] - maddpg_att_random_supervised_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('maddpg_att_random_supervised_ep_findnums_avg: {}, sigma: {}'.format(maddpg_att_random_supervised_ep_findnums_avg, sigma1))
print('maddpg_att_random_supervised_ep_rew_avg: {}, sigma: {}'.format(maddpg_att_random_supervised_ep_rew_avg, sigma2))
print('maddpg_att_random_supervised_ep_len_avg: {}, sigma: {}'.format(maddpg_att_random_supervised_ep_len_avg, sigma3))

last_steps = int(1e3)
maddpg_lstm_att_random_supervised_ep_findnums_avg = sum(maddpg_lstm_att_random_supervised_findnums_data[np.size(maddpg_lstm_att_random_supervised_findnums_data) - last_steps:np.size(maddpg_lstm_att_random_supervised_findnums_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(maddpg_lstm_att_random_supervised_findnums_data) - last_steps, np.size(maddpg_lstm_att_random_supervised_findnums_data)):
    sum1 += (maddpg_lstm_att_random_supervised_findnums_data[i] - maddpg_lstm_att_random_supervised_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
last_steps = int(1000)
maddpg_lstm_att_random_supervised_ep_rew_avg = sum(maddpg_lstm_att_random_supervised_ep_rew_mean_data[np.size(maddpg_lstm_att_random_supervised_ep_rew_mean_data) - last_steps:np.size(maddpg_lstm_att_random_supervised_ep_rew_mean_data)]) / last_steps
sum2 = 0.0
for i in range(np.size(maddpg_lstm_att_random_supervised_ep_rew_mean_data) - last_steps, np.size(maddpg_lstm_att_random_supervised_ep_rew_mean_data)):
    sum2 += (maddpg_lstm_att_random_supervised_ep_rew_mean_data[i] - maddpg_lstm_att_random_supervised_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
maddpg_lstm_att_random_supervised_ep_len_avg = sum(maddpg_lstm_att_random_supervised_ep_len_mean_data[np.size(maddpg_lstm_att_random_supervised_ep_len_mean_data) - last_steps:np.size(maddpg_lstm_att_random_supervised_ep_len_mean_data)]) / last_steps
sum3 = 0.0
for i in range(np.size(maddpg_lstm_att_random_supervised_ep_len_mean_data) - last_steps, np.size(maddpg_lstm_att_random_supervised_ep_len_mean_data)):
    sum3 += (maddpg_lstm_att_random_supervised_ep_len_mean_data[i] - maddpg_lstm_att_random_supervised_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('maddpg_lstm_att_random_supervised_ep_findnums_avg: {}, sigma: {}'.format(maddpg_lstm_att_random_supervised_ep_findnums_avg, sigma1))
print('maddpg_lstm_att_random_supervised_ep_rew_avg: {}, sigma: {}'.format(maddpg_lstm_att_random_supervised_ep_rew_avg, sigma2))
print('maddpg_lstm_att_random_supervised_ep_len_avg: {}, sigma: {}'.format(maddpg_lstm_att_random_supervised_ep_len_avg, sigma3))


plt.show()





import csv
import matplotlib.pyplot as plt
import numpy as np

td3_origin_random_findnums_data = np.loadtxt('td3_origin_random_findnums.csv', delimiter=',')
td3_random_findnums_data = np.loadtxt('td3_random_supervised_findnums.csv', delimiter=',')
td3_att_random_findnums_data = np.loadtxt('td3_att_random_supervised_findnums.csv', delimiter=',')
td3_lstm_att_random_findnums_data = np.loadtxt('td3_lstm_att_random_supervised_findnums.csv', delimiter=',')
avg_steps = 50
td3_origin_random_findnums = np.zeros((int(np.size(td3_origin_random_findnums_data) / avg_steps)))
td3_random_findnums = np.zeros((int(np.size(td3_random_findnums_data) / avg_steps)))
td3_att_random_findnums = np.zeros((int(np.size(td3_att_random_findnums_data) / avg_steps)))
td3_lstm_att_random_findnums = np.zeros((int(np.size(td3_lstm_att_random_findnums_data) / avg_steps)))
for i in range(np.size(td3_origin_random_findnums)):
    for j in range(avg_steps):
        td3_origin_random_findnums[i] += td3_origin_random_findnums_data[i * avg_steps + j]
    td3_origin_random_findnums[i] = td3_origin_random_findnums[i] / avg_steps
for i in range(np.size(td3_random_findnums)):
    for j in range(avg_steps):
        td3_random_findnums[i] += td3_random_findnums_data[i * avg_steps + j]
    td3_random_findnums[i] = td3_random_findnums[i] / avg_steps
for i in range(np.size(td3_att_random_findnums)):
    for j in range(avg_steps):
        td3_att_random_findnums[i] += td3_att_random_findnums_data[i * avg_steps + j]
    td3_att_random_findnums[i] = td3_att_random_findnums[i] / avg_steps
for i in range(np.size(td3_lstm_att_random_findnums)):
    for j in range(avg_steps):
        td3_lstm_att_random_findnums[i] += td3_lstm_att_random_findnums_data[i * avg_steps + j]
    td3_lstm_att_random_findnums[i] = td3_lstm_att_random_findnums[i] / avg_steps
x_label1 = np.arange(0, np.size(td3_random_findnums), 1)
x_label2 = np.arange(0, np.size(td3_att_random_findnums), 1)
x_label3 = np.arange(0, np.size(td3_lstm_att_random_findnums), 1)
x_label4 = np.arange(0, np.size(td3_origin_random_findnums), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), td3_random_findnums, 'b', label='td3_supervised')
plt.plot(x_label2 * 1e6 / np.size(x_label2), td3_att_random_findnums, 'g', label='td3_att_supervised')
plt.plot(x_label3 * 1e6 / np.size(x_label3), td3_lstm_att_random_findnums, 'r', label='td3_lstm_att_supervised')
plt.plot(x_label4 * 1e6 / np.size(x_label4), td3_origin_random_findnums, 'k', label='td3_origin')
plt.xlabel("time_steps")
plt.ylabel("episode_find_nums")
plt.legend()
# plt.show()

td3_origin_random_ep_rew_mean_data = np.loadtxt('td3_origin_random_ep_rew_mean.csv', delimiter=',')
td3_random_ep_rew_mean_data = np.loadtxt('td3_random_supervised_ep_rew_mean.csv', delimiter=',')
td3_random_ep_rew_mean_data_2 = np.loadtxt('td3_random_supervised_ep_rew_mean_2.csv', delimiter=',')
td3_att_random_ep_rew_mean_data = np.loadtxt('td3_att_random_supervised_ep_rew_mean.csv', delimiter=',')
td3_att_random_ep_rew_mean_data_2 = np.loadtxt('td3_att_random_supervised_ep_rew_mean_2.csv', delimiter=',')
td3_lstm_att_random_ep_rew_mean_data = np.loadtxt('td3_lstm_att_random_supervised_ep_rew_mean.csv', delimiter=',')
td3_lstm_att_random_ep_rew_mean_data_2 = np.loadtxt('td3_lstm_att_random_supervised_ep_rew_mean_2.csv', delimiter=',')
avg_steps = 50
td3_random_ep_rew_mean = np.zeros((int(np.size(td3_random_ep_rew_mean_data) / avg_steps)))
td3_att_random_ep_rew_mean = np.zeros((int(np.size(td3_att_random_ep_rew_mean_data) / avg_steps)))
td3_lstm_att_random_ep_rew_mean = np.zeros((int(np.size(td3_lstm_att_random_ep_rew_mean_data) / avg_steps)))
for i in range(np.size(td3_random_ep_rew_mean)):
    for j in range(avg_steps):
        td3_random_ep_rew_mean[i] += td3_random_ep_rew_mean_data[i * avg_steps + j]
    td3_random_ep_rew_mean[i] = td3_random_ep_rew_mean[i] / avg_steps
for i in range(np.size(td3_att_random_ep_rew_mean)):
    for j in range(avg_steps):
        td3_att_random_ep_rew_mean[i] += td3_att_random_ep_rew_mean_data[i * avg_steps + j]
    td3_att_random_ep_rew_mean[i] = td3_att_random_ep_rew_mean[i] / avg_steps
for i in range(np.size(td3_lstm_att_random_ep_rew_mean)):
    for j in range(avg_steps):
        td3_lstm_att_random_ep_rew_mean[i] += td3_lstm_att_random_ep_rew_mean_data[i * avg_steps + j]
    td3_lstm_att_random_ep_rew_mean[i] = td3_lstm_att_random_ep_rew_mean[i] / avg_steps
avg_steps = 1
td3_origin_random_ep_rew_mean = np.zeros((int(np.size(td3_origin_random_ep_rew_mean_data[:, 2]) / avg_steps)))
td3_random_ep_rew_mean_2 = np.zeros((int(np.size(td3_random_ep_rew_mean_data_2[:, 2]) / avg_steps)))
td3_att_random_ep_rew_mean_2 = np.zeros((int(np.size(td3_att_random_ep_rew_mean_data_2[:, 2]) / avg_steps)))
td3_lstm_att_random_ep_rew_mean_2 = np.zeros((int(np.size(td3_lstm_att_random_ep_rew_mean_data_2[:, 2]) / avg_steps)))
for i in range(np.size(td3_origin_random_ep_rew_mean)):
    for j in range(avg_steps):
        td3_origin_random_ep_rew_mean[i] += td3_origin_random_ep_rew_mean_data[i * avg_steps + j, 2]
    td3_origin_random_ep_rew_mean[i] = td3_origin_random_ep_rew_mean[i] / avg_steps
for i in range(np.size(td3_random_ep_rew_mean_2)):
    for j in range(avg_steps):
        td3_random_ep_rew_mean_2[i] += td3_random_ep_rew_mean_data_2[i * avg_steps + j, 2]
    td3_random_ep_rew_mean_2[i] = td3_random_ep_rew_mean_2[i] / avg_steps
for i in range(np.size(td3_att_random_ep_rew_mean_2)):
    for j in range(avg_steps):
        td3_att_random_ep_rew_mean_2[i] += td3_att_random_ep_rew_mean_data_2[i * avg_steps + j, 2]
    td3_att_random_ep_rew_mean_2[i] = td3_att_random_ep_rew_mean_2[i] / avg_steps
for i in range(np.size(td3_lstm_att_random_ep_rew_mean_2)):
    for j in range(avg_steps):
        td3_lstm_att_random_ep_rew_mean_2[i] += td3_lstm_att_random_ep_rew_mean_data_2[i * avg_steps + j, 2]
    td3_lstm_att_random_ep_rew_mean_2[i] = td3_lstm_att_random_ep_rew_mean_2[i] / avg_steps
x_label1 = np.arange(0, np.size(td3_random_ep_rew_mean) + np.size(td3_random_ep_rew_mean_2), 1)
x_label2 = np.arange(0, np.size(td3_att_random_ep_rew_mean) + np.size(td3_att_random_ep_rew_mean_2), 1)
x_label3 = np.arange(0, np.size(td3_lstm_att_random_ep_rew_mean) + np.size(td3_lstm_att_random_ep_rew_mean_2), 1)
x_label4 = np.arange(0, np.size(td3_origin_random_ep_rew_mean), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), np.hstack([td3_random_ep_rew_mean, td3_random_ep_rew_mean_2]), 'b', label='td3_supervised')
plt.plot(x_label2 * 1e6 / np.size(x_label2), np.hstack([td3_att_random_ep_rew_mean, td3_att_random_ep_rew_mean_2]), 'g', label='td3_att_supervised')
plt.plot(x_label3 * 1e6 / np.size(x_label3), np.hstack([td3_lstm_att_random_ep_rew_mean, td3_lstm_att_random_ep_rew_mean_2]), 'r', label='td3_lstm_att_supervised')
plt.plot(x_label4 * 1e6 / np.size(x_label4), td3_origin_random_ep_rew_mean, 'k', label='td3_origin')
plt.xlabel("time_steps")
plt.ylabel("ep_rew_mean")
plt.legend()
# plt.show()

td3_origin_random_ep_len_mean_data = np.loadtxt('td3_origin_random_ep_len_mean.csv', delimiter=',')
td3_random_ep_len_mean_data = np.loadtxt('td3_random_supervised_ep_len_mean.csv', delimiter=',')
td3_att_random_ep_len_mean_data = np.loadtxt('td3_att_random_supervised_ep_len_mean.csv', delimiter=',')
td3_lstm_att_random_ep_len_mean_data = np.loadtxt('td3_lstm_att_random_supervised_ep_len_mean.csv', delimiter=',')
td3_random_ep_len_mean_data_2 = np.loadtxt('td3_random_supervised_ep_len_mean_2.csv', delimiter=',')
td3_att_random_ep_len_mean_data_2 = np.loadtxt('td3_att_random_supervised_ep_len_mean_2.csv', delimiter=',')
td3_lstm_att_random_ep_len_mean_data_2 = np.loadtxt('td3_lstm_att_random_supervised_ep_len_mean_2.csv', delimiter=',')
avg_steps = 50
td3_random_ep_len_mean = np.zeros((int(np.size(td3_random_ep_len_mean_data) / avg_steps)))
td3_att_random_ep_len_mean = np.zeros((int(np.size(td3_att_random_ep_len_mean_data) / avg_steps)))
td3_lstm_att_random_ep_len_mean = np.zeros((int(np.size(td3_lstm_att_random_ep_len_mean_data) / avg_steps)))
for i in range(np.size(td3_random_ep_len_mean)):
    for j in range(avg_steps):
        td3_random_ep_len_mean[i] += td3_random_ep_len_mean_data[i * avg_steps + j]
    td3_random_ep_len_mean[i] = td3_random_ep_len_mean[i] / avg_steps
for i in range(np.size(td3_att_random_ep_len_mean)):
    for j in range(avg_steps):
        td3_att_random_ep_len_mean[i] += td3_att_random_ep_len_mean_data[i * avg_steps + j]
    td3_att_random_ep_len_mean[i] = td3_att_random_ep_len_mean[i] / avg_steps
for i in range(np.size(td3_lstm_att_random_ep_len_mean)):
    for j in range(avg_steps):
        td3_lstm_att_random_ep_len_mean[i] += td3_lstm_att_random_ep_len_mean_data[i * avg_steps + j]
    td3_lstm_att_random_ep_len_mean[i] = td3_lstm_att_random_ep_len_mean[i] / avg_steps
avg_steps = 1
td3_origin_random_ep_len_mean = np.zeros((int(np.size(td3_origin_random_ep_len_mean_data[:, 2]) / avg_steps)))
td3_random_ep_len_mean_2 = np.zeros((int(np.size(td3_random_ep_len_mean_data_2[:, 2]) / avg_steps)))
td3_att_random_ep_len_mean_2 = np.zeros((int(np.size(td3_att_random_ep_len_mean_data_2[:, 2]) / avg_steps)))
td3_lstm_att_random_ep_len_mean_2 = np.zeros((int(np.size(td3_lstm_att_random_ep_len_mean_data_2[:, 2]) / avg_steps)))
for i in range(np.size(td3_origin_random_ep_len_mean)):
    for j in range(avg_steps):
        td3_origin_random_ep_len_mean[i] += td3_origin_random_ep_len_mean_data[i * avg_steps + j, 2]
    td3_origin_random_ep_len_mean[i] = td3_origin_random_ep_len_mean[i] / avg_steps
for i in range(np.size(td3_random_ep_len_mean_2)):
    for j in range(avg_steps):
        td3_random_ep_len_mean_2[i] += td3_random_ep_len_mean_data_2[i * avg_steps + j, 2]
    td3_random_ep_len_mean_2[i] = td3_random_ep_len_mean_2[i] / avg_steps
for i in range(np.size(td3_att_random_ep_len_mean_2)):
    for j in range(avg_steps):
        td3_att_random_ep_len_mean_2[i] += td3_att_random_ep_len_mean_data_2[i * avg_steps + j, 2]
    td3_att_random_ep_len_mean_2[i] = td3_att_random_ep_len_mean_2[i] / avg_steps
for i in range(np.size(td3_lstm_att_random_ep_len_mean_2)):
    for j in range(avg_steps):
        td3_lstm_att_random_ep_len_mean_2[i] += td3_lstm_att_random_ep_len_mean_data_2[i * avg_steps + j, 2]
    td3_lstm_att_random_ep_len_mean_2[i] = td3_lstm_att_random_ep_len_mean_2[i] / avg_steps
x_label1 = np.arange(0, np.size(td3_random_ep_len_mean) + np.size(td3_random_ep_len_mean_2), 1)
x_label2 = np.arange(0, np.size(td3_att_random_ep_len_mean) + np.size(td3_att_random_ep_len_mean_2), 1)
x_label3 = np.arange(0, np.size(td3_lstm_att_random_ep_len_mean) + np.size(td3_lstm_att_random_ep_len_mean_2), 1)
x_label4 = np.arange(0, np.size(td3_origin_random_ep_len_mean), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), np.hstack([td3_random_ep_len_mean, td3_random_ep_len_mean_2]), 'b', label='td3_random')
plt.plot(x_label2 * 1e6 / np.size(x_label2), np.hstack([td3_att_random_ep_len_mean, td3_att_random_ep_len_mean_2]), 'g', label='td3_att_random')
plt.plot(x_label3 * 1e6 / np.size(x_label3), np.hstack([td3_lstm_att_random_ep_len_mean, td3_lstm_att_random_ep_len_mean_2]), 'r', label='td3_lstm_att_random')
plt.plot(x_label4 * 1e6 / np.size(x_label4), td3_origin_random_ep_len_mean, 'k', label='td3_origin')
plt.xlabel("time_steps")
plt.ylabel("ep_len_mean")
plt.legend()
# plt.show()

Agent_actor_loss_data = np.loadtxt('Agent_actor_loss.csv', delimiter=',')
Agent_att_actor_loss_data = np.loadtxt('Agent_att_actor_loss.csv', delimiter=',')
Agent_lstm_att_actor_loss_data = np.loadtxt('Agent_lstm_att_actor_loss.csv', delimiter=',')
avg_steps = 10
Agent_actor_loss = np.zeros((int(np.size(Agent_actor_loss_data) / avg_steps),))
Agent_att_actor_loss = np.zeros((int(np.size(Agent_att_actor_loss_data) / avg_steps),))
Agent_lstm_att_actor_loss = np.zeros((int(np.size(Agent_lstm_att_actor_loss_data) / avg_steps),))
for i in range(np.size(Agent_actor_loss)):
    for j in range(avg_steps):
        Agent_actor_loss[i] += Agent_actor_loss_data[i * avg_steps + j]
    Agent_actor_loss[i] = Agent_actor_loss[i] / avg_steps
for i in range(np.size(Agent_att_actor_loss)):
    for j in range(avg_steps):
        Agent_att_actor_loss[i] += Agent_att_actor_loss_data[i * avg_steps + j]
    Agent_att_actor_loss[i] = Agent_att_actor_loss[i] / avg_steps
for i in range(np.size(Agent_lstm_att_actor_loss)):
    for j in range(avg_steps):
        Agent_lstm_att_actor_loss[i] += Agent_lstm_att_actor_loss_data[i * avg_steps + j]
    Agent_lstm_att_actor_loss[i] = Agent_lstm_att_actor_loss[i] / avg_steps
x_label1 = np.arange(0, np.size(Agent_actor_loss), 1)
x_label2 = np.arange(0, np.size(Agent_att_actor_loss), 1)
x_label3 = np.arange(0, np.size(Agent_lstm_att_actor_loss), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), Agent_actor_loss, 'b', label='Agent_actor_loss')
plt.plot(x_label2 * 1e6 / np.size(x_label2), Agent_att_actor_loss, 'g', label='Agent_att_actor_loss')
plt.plot(x_label3 * 1e6 / np.size(x_label3), Agent_lstm_att_actor_loss, 'r', label='Agent_lstm_att_actor_loss')
plt.xlabel("time_steps")
plt.ylabel("actor_loss")
plt.legend()
# plt.show()

Agent_critic_loss_data = np.loadtxt('Agent_critic_loss.csv', delimiter=',')
Agent_att_critic_loss_data = np.loadtxt('Agent_att_critic_loss.csv', delimiter=',')
Agent_lstm_att_critic_loss_data = np.loadtxt('Agent_lstm_att_critic_loss.csv', delimiter=',')
avg_steps = 30
Agent_critic_loss = np.zeros((int(np.size(Agent_critic_loss_data) / avg_steps),))
Agent_att_critic_loss = np.zeros((int(np.size(Agent_att_critic_loss_data) / avg_steps),))
Agent_lstm_att_critic_loss = np.zeros((int(np.size(Agent_lstm_att_critic_loss_data) / avg_steps),))
for i in range(np.size(Agent_critic_loss)):
    for j in range(avg_steps):
        Agent_critic_loss[i] += Agent_critic_loss_data[i * avg_steps + j]
    Agent_critic_loss[i] = Agent_critic_loss[i] / avg_steps
for i in range(np.size(Agent_att_critic_loss)):
    for j in range(avg_steps):
        Agent_att_critic_loss[i] += Agent_att_critic_loss_data[i * avg_steps + j]
    Agent_att_critic_loss[i] = Agent_att_critic_loss[i] / avg_steps
for i in range(np.size(Agent_lstm_att_critic_loss)):
    for j in range(avg_steps):
        Agent_lstm_att_critic_loss[i] += Agent_lstm_att_critic_loss_data[i * avg_steps + j]
    Agent_lstm_att_critic_loss[i] = Agent_lstm_att_critic_loss[i] / avg_steps
x_label1 = np.arange(0, np.size(Agent_critic_loss), 1)
x_label2 = np.arange(0, np.size(Agent_att_critic_loss), 1)
x_label3 = np.arange(0, np.size(Agent_lstm_att_critic_loss), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), Agent_critic_loss, 'b', label='Agent_critic_loss')
plt.plot(x_label2 * 1e6 / np.size(x_label2), Agent_att_critic_loss, 'g', label='Agent_att_critic_loss')
plt.plot(x_label3 * 1e6 / np.size(x_label3), Agent_lstm_att_critic_loss, 'r', label='Agent_lstm_att_critic_loss')
plt.xlabel("time_steps")
plt.ylabel("critic_loss")
plt.legend()

last_steps = int(1e4)
td3_random_ep_findnums_avg = sum(td3_random_findnums_data[np.size(td3_random_findnums_data) - last_steps:np.size(td3_random_findnums_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(td3_random_findnums_data) - last_steps, np.size(td3_random_findnums_data)):
    sum1 += (td3_random_findnums_data[i] - td3_random_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
last_steps = int(100)
td3_random_ep_rew_avg = sum(td3_random_ep_rew_mean_data_2[900:1000, 2]) / last_steps
sum2 = 0.0
for i in range(np.size(td3_random_ep_rew_mean_data_2[:, 2]) - last_steps, np.size(td3_random_ep_rew_mean_data_2[:, 2])):
    sum2 += (td3_random_ep_rew_mean_data_2[i, 2] - td3_random_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
td3_random_ep_len_avg = sum(td3_random_ep_len_mean_data_2[900:1000, 2]) / last_steps
sum3 = 0.0
for i in range(np.size(td3_random_ep_len_mean_data_2[:, 2]) - last_steps, np.size(td3_random_ep_len_mean_data_2[:, 2])):
    sum3 += (td3_random_ep_len_mean_data_2[i, 2] - td3_random_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('td3_random_ep_findnums_avg: {}, sigma: {}'.format(td3_random_ep_findnums_avg, sigma1))
print('td3_random_ep_rew_avg: {}, sigma: {}'.format(td3_random_ep_rew_avg, sigma2))
print('td3_random_ep_len_avg: {}, sigma: {}'.format(td3_random_ep_len_avg, sigma3))

last_steps = int(1e4)
td3_att_random_ep_findnums_avg = sum(td3_att_random_findnums_data[np.size(td3_att_random_findnums_data) - last_steps:np.size(td3_att_random_findnums_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(td3_att_random_findnums_data) - last_steps, np.size(td3_att_random_findnums_data)):
    sum1 += (td3_att_random_findnums_data[i] - td3_att_random_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
last_steps = int(100)
td3_att_random_ep_rew_avg = sum(td3_att_random_ep_rew_mean_data_2[900:1000, 2]) / last_steps
sum2 = 0.0
for i in range(np.size(td3_att_random_ep_rew_mean_data_2[:, 2]) - last_steps, np.size(td3_att_random_ep_rew_mean_data_2[:, 2])):
    sum2 += (td3_att_random_ep_rew_mean_data_2[i, 2] - td3_att_random_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
td3_att_random_ep_len_avg = sum(td3_att_random_ep_len_mean_data_2[900:1000, 2]) / last_steps
sum3 = 0.0
for i in range(np.size(td3_att_random_ep_len_mean_data_2[:, 2]) - last_steps, np.size(td3_att_random_ep_len_mean_data_2[:, 2])):
    sum3 += (td3_att_random_ep_len_mean_data_2[i, 2] - td3_att_random_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('td3_att_random_ep_findnums_avg: {}, sigma: {}'.format(td3_att_random_ep_findnums_avg, sigma1))
print('td3_att_random_ep_rew_avg: {}, sigma: {}'.format(td3_att_random_ep_rew_avg, sigma2))
print('td3_att_random_ep_len_avg: {}, sigma: {}'.format(td3_att_random_ep_len_avg, sigma3))

last_steps = int(1e4)
td3_lstm_att_random_ep_findnums_avg = sum(td3_lstm_att_random_findnums_data[np.size(td3_lstm_att_random_findnums_data) - last_steps:np.size(td3_lstm_att_random_findnums_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(td3_lstm_att_random_findnums_data) - last_steps, np.size(td3_lstm_att_random_findnums_data)):
    sum1 += (td3_lstm_att_random_findnums_data[i] - td3_lstm_att_random_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
last_steps = int(100)
td3_lstm_att_random_ep_rew_avg = sum(td3_lstm_att_random_ep_rew_mean_data_2[900:1000, 2]) / last_steps
sum2 = 0.0
for i in range(np.size(td3_lstm_att_random_ep_rew_mean_data_2[:, 2]) - last_steps, np.size(td3_lstm_att_random_ep_rew_mean_data_2[:, 2])):
    sum2 += (td3_lstm_att_random_ep_rew_mean_data_2[i, 2] - td3_lstm_att_random_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
td3_lstm_att_random_ep_len_avg = sum(td3_lstm_att_random_ep_len_mean_data_2[900:1000, 2]) / last_steps
sum3 = 0.0
for i in range(np.size(td3_lstm_att_random_ep_len_mean_data_2[:, 2]) - last_steps, np.size(td3_lstm_att_random_ep_len_mean_data_2[:, 2])):
    sum3 += (td3_lstm_att_random_ep_len_mean_data_2[i, 2] - td3_lstm_att_random_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('td3_lstm_att_random_ep_findnums_avg: {}, sigma: {}'.format(td3_lstm_att_random_ep_findnums_avg, sigma1))
print('td3_lstm_att_random_ep_rew_avg: {}, sigma: {}'.format(td3_lstm_att_random_ep_rew_avg, sigma2))
print('td3_lstm_att_random_ep_len_avg: {}, sigma: {}'.format(td3_lstm_att_random_ep_len_avg, sigma3))

plt.show()





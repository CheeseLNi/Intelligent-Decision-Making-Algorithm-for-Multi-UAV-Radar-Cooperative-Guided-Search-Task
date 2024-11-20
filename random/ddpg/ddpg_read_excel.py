import csv
import matplotlib.pyplot as plt
import numpy as np

ddpg_origin_random_findnums_data = np.loadtxt('ddpg_origin_random_findnums.csv', delimiter=',')
ddpg_random_findnums_data = np.loadtxt('ddpg_random_supervised_findnums.csv', delimiter=',')
ddpg_att_random_findnums_data = np.loadtxt('ddpg_att_random_supervised_findnums.csv', delimiter=',')
ddpg_lstm_att_random_findnums_data = np.loadtxt('ddpg_lstm_att_random_supervised_findnums.csv', delimiter=',')
avg_steps = 50
ddpg_origin_random_findnums = np.zeros((int(np.size(ddpg_origin_random_findnums_data) / avg_steps)))
ddpg_random_findnums = np.zeros((int(np.size(ddpg_random_findnums_data) / avg_steps)))
ddpg_att_random_findnums = np.zeros((int(np.size(ddpg_att_random_findnums_data) / avg_steps)))
ddpg_lstm_att_random_findnums = np.zeros((int(np.size(ddpg_lstm_att_random_findnums_data) / avg_steps)))
for i in range(np.size(ddpg_origin_random_findnums)):
    for j in range(avg_steps):
        ddpg_origin_random_findnums[i] += ddpg_origin_random_findnums_data[i * avg_steps + j]
    ddpg_origin_random_findnums[i] = ddpg_origin_random_findnums[i] / avg_steps
for i in range(np.size(ddpg_random_findnums)):
    for j in range(avg_steps):
        ddpg_random_findnums[i] += ddpg_random_findnums_data[i * avg_steps + j]
    ddpg_random_findnums[i] = ddpg_random_findnums[i] / avg_steps
for i in range(np.size(ddpg_att_random_findnums)):
    for j in range(avg_steps):
        ddpg_att_random_findnums[i] += ddpg_att_random_findnums_data[i * avg_steps + j]
    ddpg_att_random_findnums[i] = ddpg_att_random_findnums[i] / avg_steps
for i in range(np.size(ddpg_lstm_att_random_findnums)):
    for j in range(avg_steps):
        ddpg_lstm_att_random_findnums[i] += ddpg_lstm_att_random_findnums_data[i * avg_steps + j]
    ddpg_lstm_att_random_findnums[i] = ddpg_lstm_att_random_findnums[i] / avg_steps
x_label1 = np.arange(0, np.size(ddpg_random_findnums), 1)
x_label2 = np.arange(0, np.size(ddpg_att_random_findnums), 1)
x_label3 = np.arange(0, np.size(ddpg_lstm_att_random_findnums), 1)
x_label4 = np.arange(0, np.size(ddpg_origin_random_findnums), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), ddpg_random_findnums, 'b', label='ddpg_supervised')
plt.plot(x_label2 * 1e6 / np.size(x_label2), ddpg_att_random_findnums, 'g', label='ddpg_att_supervised')
plt.plot(x_label3 * 1e6 / np.size(x_label3), ddpg_lstm_att_random_findnums, 'r', label='ddpg_lstm_att_supervised')
plt.plot(x_label4 * 1e6 / np.size(x_label4), ddpg_origin_random_findnums, 'k', label='ddpg_origin')
plt.xlabel("time_steps")
plt.ylabel("episode_find_nums")
plt.legend()
# plt.show()

ddpg_origin_random_ep_rew_mean_data = np.loadtxt('ddpg_origin_random_ep_rew_mean.csv', delimiter=',')
ddpg_random_ep_rew_mean_data = np.loadtxt('ddpg_random_supervised_ep_rew_mean.csv', delimiter=',')
ddpg_random_ep_rew_mean_data_2 = np.loadtxt('ddpg_random_supervised_ep_rew_mean_2.csv', delimiter=',')
ddpg_att_random_ep_rew_mean_data = np.loadtxt('ddpg_att_random_supervised_ep_rew_mean.csv', delimiter=',')
ddpg_att_random_ep_rew_mean_data_2 = np.loadtxt('ddpg_att_random_supervised_ep_rew_mean_2.csv', delimiter=',')
ddpg_lstm_att_random_ep_rew_mean_data = np.loadtxt('ddpg_lstm_att_random_supervised_ep_rew_mean.csv', delimiter=',')
ddpg_lstm_att_random_ep_rew_mean_data_2 = np.loadtxt('ddpg_lstm_att_random_supervised_ep_rew_mean_2.csv', delimiter=',')
avg_steps = 50
ddpg_random_ep_rew_mean = np.zeros((int(np.size(ddpg_random_ep_rew_mean_data) / avg_steps)))
ddpg_att_random_ep_rew_mean = np.zeros((int(np.size(ddpg_att_random_ep_rew_mean_data) / avg_steps)))
ddpg_lstm_att_random_ep_rew_mean = np.zeros((int(np.size(ddpg_lstm_att_random_ep_rew_mean_data) / avg_steps)))
for i in range(np.size(ddpg_random_ep_rew_mean)):
    for j in range(avg_steps):
        ddpg_random_ep_rew_mean[i] += ddpg_random_ep_rew_mean_data[i * avg_steps + j]
    ddpg_random_ep_rew_mean[i] = ddpg_random_ep_rew_mean[i] / avg_steps
for i in range(np.size(ddpg_att_random_ep_rew_mean)):
    for j in range(avg_steps):
        ddpg_att_random_ep_rew_mean[i] += ddpg_att_random_ep_rew_mean_data[i * avg_steps + j]
    ddpg_att_random_ep_rew_mean[i] = ddpg_att_random_ep_rew_mean[i] / avg_steps
for i in range(np.size(ddpg_lstm_att_random_ep_rew_mean)):
    for j in range(avg_steps):
        ddpg_lstm_att_random_ep_rew_mean[i] += ddpg_lstm_att_random_ep_rew_mean_data[i * avg_steps + j]
    ddpg_lstm_att_random_ep_rew_mean[i] = ddpg_lstm_att_random_ep_rew_mean[i] / avg_steps
avg_steps = 1
ddpg_origin_random_ep_rew_mean = np.zeros((int(np.size(ddpg_origin_random_ep_rew_mean_data[:, 2]) / avg_steps)))
ddpg_random_ep_rew_mean_2 = np.zeros((int(np.size(ddpg_random_ep_rew_mean_data_2[:, 2]) / avg_steps)))
ddpg_att_random_ep_rew_mean_2 = np.zeros((int(np.size(ddpg_att_random_ep_rew_mean_data_2[:, 2]) / avg_steps)))
ddpg_lstm_att_random_ep_rew_mean_2 = np.zeros((int(np.size(ddpg_lstm_att_random_ep_rew_mean_data_2[:, 2]) / avg_steps)))
for i in range(np.size(ddpg_origin_random_ep_rew_mean)):
    for j in range(avg_steps):
        ddpg_origin_random_ep_rew_mean[i] += ddpg_origin_random_ep_rew_mean_data[i * avg_steps + j, 2]
    ddpg_origin_random_ep_rew_mean[i] = ddpg_origin_random_ep_rew_mean[i] / avg_steps
for i in range(np.size(ddpg_random_ep_rew_mean_2)):
    for j in range(avg_steps):
        ddpg_random_ep_rew_mean_2[i] += ddpg_random_ep_rew_mean_data_2[i * avg_steps + j, 2]
    ddpg_random_ep_rew_mean_2[i] = ddpg_random_ep_rew_mean_2[i] / avg_steps
for i in range(np.size(ddpg_att_random_ep_rew_mean_2)):
    for j in range(avg_steps):
        ddpg_att_random_ep_rew_mean_2[i] += ddpg_att_random_ep_rew_mean_data_2[i * avg_steps + j, 2]
    ddpg_att_random_ep_rew_mean_2[i] = ddpg_att_random_ep_rew_mean_2[i] / avg_steps
for i in range(np.size(ddpg_lstm_att_random_ep_rew_mean_2)):
    for j in range(avg_steps):
        ddpg_lstm_att_random_ep_rew_mean_2[i] += ddpg_lstm_att_random_ep_rew_mean_data_2[i * avg_steps + j, 2]
    ddpg_lstm_att_random_ep_rew_mean_2[i] = ddpg_lstm_att_random_ep_rew_mean_2[i] / avg_steps
x_label1 = np.arange(0, np.size(ddpg_random_ep_rew_mean) + np.size(ddpg_random_ep_rew_mean_2), 1)
x_label2 = np.arange(0, np.size(ddpg_att_random_ep_rew_mean) + np.size(ddpg_att_random_ep_rew_mean_2), 1)
x_label3 = np.arange(0, np.size(ddpg_lstm_att_random_ep_rew_mean) + np.size(ddpg_lstm_att_random_ep_rew_mean_2), 1)
x_label4 = np.arange(0, np.size(ddpg_origin_random_ep_rew_mean), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), np.hstack([ddpg_random_ep_rew_mean, ddpg_random_ep_rew_mean_2]), 'b', label='ddpg_supervised')
plt.plot(x_label2 * 1e6 / np.size(x_label2), np.hstack([ddpg_att_random_ep_rew_mean, ddpg_att_random_ep_rew_mean_2]), 'g', label='ddpg_att_supervised')
plt.plot(x_label3 * 1e6 / np.size(x_label3), np.hstack([ddpg_lstm_att_random_ep_rew_mean, ddpg_lstm_att_random_ep_rew_mean_2]), 'r', label='ddpg_lstm_att_supervised')
plt.plot(x_label4 * 1e6 / np.size(x_label4), ddpg_origin_random_ep_rew_mean, 'k', label='ddpg_origin')
plt.xlabel("time_steps")
plt.ylabel("ep_rew_mean")
plt.legend()
# plt.show()

ddpg_origin_random_ep_len_mean_data = np.loadtxt('ddpg_origin_random_ep_len_mean.csv', delimiter=',')
ddpg_random_ep_len_mean_data = np.loadtxt('ddpg_random_supervised_ep_len_mean.csv', delimiter=',')
ddpg_att_random_ep_len_mean_data = np.loadtxt('ddpg_att_random_supervised_ep_len_mean.csv', delimiter=',')
ddpg_lstm_att_random_ep_len_mean_data = np.loadtxt('ddpg_lstm_att_random_supervised_ep_len_mean.csv', delimiter=',')
ddpg_random_ep_len_mean_data_2 = np.loadtxt('ddpg_random_supervised_ep_len_mean_2.csv', delimiter=',')
ddpg_att_random_ep_len_mean_data_2 = np.loadtxt('ddpg_att_random_supervised_ep_len_mean_2.csv', delimiter=',')
ddpg_lstm_att_random_ep_len_mean_data_2 = np.loadtxt('ddpg_lstm_att_random_supervised_ep_len_mean_2.csv', delimiter=',')
avg_steps = 50
ddpg_random_ep_len_mean = np.zeros((int(np.size(ddpg_random_ep_len_mean_data) / avg_steps)))
ddpg_att_random_ep_len_mean = np.zeros((int(np.size(ddpg_att_random_ep_len_mean_data) / avg_steps)))
ddpg_lstm_att_random_ep_len_mean = np.zeros((int(np.size(ddpg_lstm_att_random_ep_len_mean_data) / avg_steps)))
for i in range(np.size(ddpg_random_ep_len_mean)):
    for j in range(avg_steps):
        ddpg_random_ep_len_mean[i] += ddpg_random_ep_len_mean_data[i * avg_steps + j]
    ddpg_random_ep_len_mean[i] = ddpg_random_ep_len_mean[i] / avg_steps
for i in range(np.size(ddpg_att_random_ep_len_mean)):
    for j in range(avg_steps):
        ddpg_att_random_ep_len_mean[i] += ddpg_att_random_ep_len_mean_data[i * avg_steps + j]
    ddpg_att_random_ep_len_mean[i] = ddpg_att_random_ep_len_mean[i] / avg_steps
for i in range(np.size(ddpg_lstm_att_random_ep_len_mean)):
    for j in range(avg_steps):
        ddpg_lstm_att_random_ep_len_mean[i] += ddpg_lstm_att_random_ep_len_mean_data[i * avg_steps + j]
    ddpg_lstm_att_random_ep_len_mean[i] = ddpg_lstm_att_random_ep_len_mean[i] / avg_steps
avg_steps = 1
ddpg_origin_random_ep_len_mean = np.zeros((int(np.size(ddpg_origin_random_ep_len_mean_data[:, 2]) / avg_steps)))
ddpg_random_ep_len_mean_2 = np.zeros((int(np.size(ddpg_random_ep_len_mean_data_2[:, 2]) / avg_steps)))
ddpg_att_random_ep_len_mean_2 = np.zeros((int(np.size(ddpg_att_random_ep_len_mean_data_2[:, 2]) / avg_steps)))
ddpg_lstm_att_random_ep_len_mean_2 = np.zeros((int(np.size(ddpg_lstm_att_random_ep_len_mean_data_2[:, 2]) / avg_steps)))
for i in range(np.size(ddpg_origin_random_ep_len_mean)):
    for j in range(avg_steps):
        ddpg_origin_random_ep_len_mean[i] += ddpg_origin_random_ep_len_mean_data[i * avg_steps + j, 2]
    ddpg_origin_random_ep_len_mean[i] = ddpg_origin_random_ep_len_mean[i] / avg_steps
for i in range(np.size(ddpg_random_ep_len_mean_2)):
    for j in range(avg_steps):
        ddpg_random_ep_len_mean_2[i] += ddpg_random_ep_len_mean_data_2[i * avg_steps + j, 2]
    ddpg_random_ep_len_mean_2[i] = ddpg_random_ep_len_mean_2[i] / avg_steps
for i in range(np.size(ddpg_att_random_ep_len_mean_2)):
    for j in range(avg_steps):
        ddpg_att_random_ep_len_mean_2[i] += ddpg_att_random_ep_len_mean_data_2[i * avg_steps + j, 2]
    ddpg_att_random_ep_len_mean_2[i] = ddpg_att_random_ep_len_mean_2[i] / avg_steps
for i in range(np.size(ddpg_lstm_att_random_ep_len_mean_2)):
    for j in range(avg_steps):
        ddpg_lstm_att_random_ep_len_mean_2[i] += ddpg_lstm_att_random_ep_len_mean_data_2[i * avg_steps + j, 2]
    ddpg_lstm_att_random_ep_len_mean_2[i] = ddpg_lstm_att_random_ep_len_mean_2[i] / avg_steps
x_label1 = np.arange(0, np.size(ddpg_random_ep_len_mean) + np.size(ddpg_random_ep_len_mean_2), 1)
x_label2 = np.arange(0, np.size(ddpg_att_random_ep_len_mean) + np.size(ddpg_att_random_ep_len_mean_2), 1)
x_label3 = np.arange(0, np.size(ddpg_lstm_att_random_ep_len_mean) + np.size(ddpg_lstm_att_random_ep_len_mean_2), 1)
x_label4 = np.arange(0, np.size(ddpg_origin_random_ep_len_mean), 1)
plt.figure()
plt.plot(x_label1 * 1e6 / np.size(x_label1), np.hstack([ddpg_random_ep_len_mean, ddpg_random_ep_len_mean_2]), 'b', label='ddpg_random')
plt.plot(x_label2 * 1e6 / np.size(x_label2), np.hstack([ddpg_att_random_ep_len_mean, ddpg_att_random_ep_len_mean_2]), 'g', label='ddpg_att_random')
plt.plot(x_label3 * 1e6 / np.size(x_label3), np.hstack([ddpg_lstm_att_random_ep_len_mean, ddpg_lstm_att_random_ep_len_mean_2]), 'r', label='ddpg_lstm_att_random')
plt.plot(x_label4 * 1e6 / np.size(x_label4), ddpg_origin_random_ep_len_mean, 'k', label='ddpg_origin')
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
ddpg_random_ep_findnums_avg = sum(ddpg_random_findnums_data[np.size(ddpg_random_findnums_data) - last_steps:np.size(ddpg_random_findnums_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(ddpg_random_findnums_data) - last_steps, np.size(ddpg_random_findnums_data)):
    sum1 += (ddpg_random_findnums_data[i] - ddpg_random_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
last_steps = int(100)
ddpg_random_ep_rew_avg = sum(ddpg_random_ep_rew_mean_data_2[900:1000, 2]) / last_steps
sum2 = 0.0
for i in range(np.size(ddpg_random_ep_rew_mean_data_2[:, 2]) - last_steps, np.size(ddpg_random_ep_rew_mean_data_2[:, 2])):
    sum2 += (ddpg_random_ep_rew_mean_data_2[i, 2] - ddpg_random_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
ddpg_random_ep_len_avg = sum(ddpg_random_ep_len_mean_data_2[900:1000, 2]) / last_steps
sum3 = 0.0
for i in range(np.size(ddpg_random_ep_len_mean_data_2[:, 2]) - last_steps, np.size(ddpg_random_ep_len_mean_data_2[:, 2])):
    sum3 += (ddpg_random_ep_len_mean_data_2[i, 2] - ddpg_random_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('ddpg_random_ep_findnums_avg: {}, sigma: {}'.format(ddpg_random_ep_findnums_avg, sigma1))
print('ddpg_random_ep_rew_avg: {}, sigma: {}'.format(ddpg_random_ep_rew_avg, sigma2))
print('ddpg_random_ep_len_avg: {}, sigma: {}'.format(ddpg_random_ep_len_avg, sigma3))

last_steps = int(1e4)
ddpg_att_random_ep_findnums_avg = sum(ddpg_att_random_findnums_data[np.size(ddpg_att_random_findnums_data) - last_steps:np.size(ddpg_att_random_findnums_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(ddpg_att_random_findnums_data) - last_steps, np.size(ddpg_att_random_findnums_data)):
    sum1 += (ddpg_att_random_findnums_data[i] - ddpg_att_random_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
last_steps = int(100)
ddpg_att_random_ep_rew_avg = sum(ddpg_att_random_ep_rew_mean_data_2[900:1000, 2]) / last_steps
sum2 = 0.0
for i in range(np.size(ddpg_att_random_ep_rew_mean_data_2[:, 2]) - last_steps, np.size(ddpg_att_random_ep_rew_mean_data_2[:, 2])):
    sum2 += (ddpg_att_random_ep_rew_mean_data_2[i, 2] - ddpg_att_random_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
ddpg_att_random_ep_len_avg = sum(ddpg_att_random_ep_len_mean_data_2[900:1000, 2]) / last_steps
sum3 = 0.0
for i in range(np.size(ddpg_att_random_ep_len_mean_data_2[:, 2]) - last_steps, np.size(ddpg_att_random_ep_len_mean_data_2[:, 2])):
    sum3 += (ddpg_att_random_ep_len_mean_data_2[i, 2] - ddpg_att_random_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('ddpg_att_random_ep_findnums_avg: {}, sigma: {}'.format(ddpg_att_random_ep_findnums_avg, sigma1))
print('ddpg_att_random_ep_rew_avg: {}, sigma: {}'.format(ddpg_att_random_ep_rew_avg, sigma2))
print('ddpg_att_random_ep_len_avg: {}, sigma: {}'.format(ddpg_att_random_ep_len_avg, sigma3))

last_steps = int(1e4)
ddpg_lstm_att_random_ep_findnums_avg = sum(ddpg_lstm_att_random_findnums_data[np.size(ddpg_lstm_att_random_findnums_data) - last_steps:np.size(ddpg_lstm_att_random_findnums_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(ddpg_lstm_att_random_findnums_data) - last_steps, np.size(ddpg_lstm_att_random_findnums_data)):
    sum1 += (ddpg_lstm_att_random_findnums_data[i] - ddpg_lstm_att_random_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
last_steps = int(100)
ddpg_lstm_att_random_ep_rew_avg = sum(ddpg_lstm_att_random_ep_rew_mean_data_2[900:1000, 2]) / last_steps
sum2 = 0.0
for i in range(np.size(ddpg_lstm_att_random_ep_rew_mean_data_2[:, 2]) - last_steps, np.size(ddpg_lstm_att_random_ep_rew_mean_data_2[:, 2])):
    sum2 += (ddpg_lstm_att_random_ep_rew_mean_data_2[i, 2] - ddpg_lstm_att_random_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
ddpg_lstm_att_random_ep_len_avg = sum(ddpg_lstm_att_random_ep_len_mean_data_2[900:1000, 2]) / last_steps
sum3 = 0.0
for i in range(np.size(ddpg_lstm_att_random_ep_len_mean_data_2[:, 2]) - last_steps, np.size(ddpg_lstm_att_random_ep_len_mean_data_2[:, 2])):
    sum3 += (ddpg_lstm_att_random_ep_len_mean_data_2[i, 2] - ddpg_lstm_att_random_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('ddpg_lstm_att_random_ep_findnums_avg: {}, sigma: {}'.format(ddpg_lstm_att_random_ep_findnums_avg, sigma1))
print('ddpg_lstm_att_random_ep_rew_avg: {}, sigma: {}'.format(ddpg_lstm_att_random_ep_rew_avg, sigma2))
print('ddpg_lstm_att_random_ep_len_avg: {}, sigma: {}'.format(ddpg_lstm_att_random_ep_len_avg, sigma3))



plt.show()





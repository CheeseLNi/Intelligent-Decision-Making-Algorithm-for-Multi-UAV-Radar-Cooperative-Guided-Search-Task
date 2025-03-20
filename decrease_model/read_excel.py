import matplotlib.pyplot as plt
import numpy as np

ppo_fix_findnums_data = np.loadtxt('./fix/maddpg/ppo_fix_findnums.csv', delimiter=',')
maddpg_fix_findnums_data = np.loadtxt('./fix/maddpg/maddpg_fix_findnums.csv', delimiter=',')
maddpg_att_fix_findnums_data = np.loadtxt('./fix/maddpg/maddpg_att_fix_findnums.csv', delimiter=',')
maddpg_lstm_att_fix_findnums_data = np.loadtxt('./fix/maddpg/maddpg_lstm_att_fix_findnums.csv', delimiter=',')

avg_steps = 100
ppo_fix_findnums = np.zeros((int(np.size(ppo_fix_findnums_data) / avg_steps),))
for i in range(np.size(ppo_fix_findnums)):
    for j in range(avg_steps):
        ppo_fix_findnums[i] += ppo_fix_findnums_data[i * avg_steps + j]
    ppo_fix_findnums[i] = ppo_fix_findnums[i] / avg_steps
avg_steps = 20
maddpg_fix_findnums = np.zeros((int(np.size(maddpg_fix_findnums_data) / avg_steps),))
for i in range(np.size(maddpg_fix_findnums)):
    for j in range(avg_steps):
        maddpg_fix_findnums[i] += maddpg_fix_findnums_data[i * avg_steps + j]
    maddpg_fix_findnums[i] = maddpg_fix_findnums[i] / avg_steps
maddpg_att_fix_findnums = np.zeros((int(np.size(maddpg_att_fix_findnums_data) / avg_steps),))
for i in range(np.size(maddpg_att_fix_findnums)):
    for j in range(avg_steps):
        maddpg_att_fix_findnums[i] += maddpg_att_fix_findnums_data[i * avg_steps + j]
    maddpg_att_fix_findnums[i] = maddpg_att_fix_findnums[i] / avg_steps
maddpg_lstm_att_fix_findnums = np.zeros((int(np.size(maddpg_att_fix_findnums_data) / avg_steps),))
for i in range(np.size(maddpg_lstm_att_fix_findnums)):
    for j in range(avg_steps):
        maddpg_lstm_att_fix_findnums[i] += maddpg_lstm_att_fix_findnums_data[i * avg_steps + j]
    maddpg_lstm_att_fix_findnums[i] = maddpg_lstm_att_fix_findnums[i] / avg_steps

x_label1 = np.arange(0, np.size(ppo_fix_findnums), 1)
x_label2 = np.arange(0, np.size(maddpg_fix_findnums), 1)
x_label3 = np.arange(0, np.size(maddpg_att_fix_findnums), 1)
x_label4 = np.arange(0, np.size(maddpg_lstm_att_fix_findnums), 1)

plt.figure()
# plt.plot(x_label1 * 1e6 / np.size(x_label1), ppo_fix_findnums, 'b', label='ppo')
plt.plot(x_label2 * 1e6 / np.size(x_label2), maddpg_fix_findnums, 'b', label='maddpg')
plt.plot(x_label3 * 1e6 / np.size(x_label3), maddpg_att_fix_findnums, 'g', label='maddpg_att')
plt.plot(x_label4 * 1e6 / np.size(x_label4), maddpg_lstm_att_fix_findnums, 'r', label='maddpg_lstm_att')
plt.xlabel("time_steps")
plt.ylabel("episode_find_nums")
plt.title("fix_environment")
plt.legend()
# plt.show()

ppo_fix_ep_rew_data = np.loadtxt('./fix/maddpg/ppo_fix_ep_rew_mean.csv', delimiter=',')
maddpg_fix_ep_rew_data = np.loadtxt('./fix/maddpg/maddpg_fix_ep_rew_mean.csv', delimiter=',')
maddpg_att_fix_ep_rew_data = np.loadtxt('./fix/maddpg/maddpg_att_fix_ep_rew_mean.csv', delimiter=',')
maddpg_lstm_att_fix_ep_rew_data = np.loadtxt('./fix/maddpg/maddpg_lstm_att_fix_ep_rew_mean.csv', delimiter=',')

avg_steps = 1
ppo_fix_ep_rew = np.zeros((int(np.size(ppo_fix_ep_rew_data[:, 2]) / avg_steps),))
for i in range(np.size(ppo_fix_ep_rew)):
    for j in range(avg_steps):
        ppo_fix_ep_rew[i] += ppo_fix_ep_rew_data[i * avg_steps + j, 2]
    ppo_fix_ep_rew[i] = ppo_fix_ep_rew[i] / avg_steps
avg_steps = 20
maddpg_fix_ep_rew = np.zeros((int(np.size(maddpg_fix_ep_rew_data) / avg_steps),))
for i in range(np.size(maddpg_fix_ep_rew)):
    for j in range(avg_steps):
        maddpg_fix_ep_rew[i] += maddpg_fix_ep_rew_data[i * avg_steps + j]
    maddpg_fix_ep_rew[i] = maddpg_fix_ep_rew[i] / avg_steps
maddpg_att_fix_ep_rew = np.zeros((int(np.size(maddpg_att_fix_ep_rew_data) / avg_steps),))
for i in range(np.size(maddpg_att_fix_ep_rew)):
    for j in range(avg_steps):
        maddpg_att_fix_ep_rew[i] += maddpg_att_fix_ep_rew_data[i * avg_steps + j]
    maddpg_att_fix_ep_rew[i] = maddpg_att_fix_ep_rew[i] / avg_steps
maddpg_lstm_att_fix_ep_rew = np.zeros((int(np.size(maddpg_lstm_att_fix_ep_rew_data) / avg_steps),))
for i in range(np.size(maddpg_lstm_att_fix_ep_rew)):
    for j in range(avg_steps):
        maddpg_lstm_att_fix_ep_rew[i] += maddpg_lstm_att_fix_ep_rew_data[i * avg_steps + j]
    maddpg_lstm_att_fix_ep_rew[i] = maddpg_lstm_att_fix_ep_rew[i] / avg_steps

x_label1 = np.arange(0, np.size(ppo_fix_ep_rew), 1)
x_label2 = np.arange(0, np.size(maddpg_fix_ep_rew), 1)
x_label3 = np.arange(0, np.size(maddpg_att_fix_ep_rew), 1)
x_label4 = np.arange(0, np.size(maddpg_lstm_att_fix_ep_rew), 1)
plt.figure()
# plt.plot(x_label1 * 1e6 / np.size(x_label1), ppo_fix_ep_rew, 'b', label='ppo')
plt.plot(x_label2 * 1e6 / np.size(x_label2), maddpg_fix_ep_rew, 'b', label='maddpg')
plt.plot(x_label3 * 1e6 / np.size(x_label3), maddpg_att_fix_ep_rew, 'g', label='maddpg_att')
plt.plot(x_label4 * 1e6 / np.size(x_label3), maddpg_lstm_att_fix_ep_rew, 'r', label='maddpg_lstm_att')
plt.xlabel("time_steps")
plt.ylabel("ep_rew_mean")
plt.title("fix_environment")
plt.legend()
# plt.show()

ppo_fix_ep_len_data = np.loadtxt('./fix/maddpg/ppo_fix_ep_len_mean.csv', delimiter=',')
maddpg_fix_ep_len_data = np.loadtxt('./fix/maddpg/maddpg_fix_ep_len_mean.csv', delimiter=',')
maddpg_att_fix_ep_len_data = np.loadtxt('./fix/maddpg/maddpg_att_fix_ep_len_mean.csv', delimiter=',')
maddpg_lstm_att_fix_ep_len_data = np.loadtxt('./fix/maddpg/maddpg_lstm_att_fix_ep_len_mean.csv', delimiter=',')

avg_steps = 1
ppo_fix_ep_len = np.zeros((int(np.size(ppo_fix_ep_len_data[:, 2]) / avg_steps),))
for i in range(np.size(ppo_fix_ep_len)):
    for j in range(avg_steps):
        ppo_fix_ep_len[i] += ppo_fix_ep_len_data[i * avg_steps + j, 2]
    ppo_fix_ep_len[i] = ppo_fix_ep_len[i] / avg_steps
avg_steps = 20
maddpg_fix_ep_len = np.zeros((int(np.size(maddpg_fix_ep_len_data) / avg_steps),))
for i in range(np.size(maddpg_fix_ep_len)):
    for j in range(avg_steps):
        maddpg_fix_ep_len[i] += maddpg_fix_ep_len_data[i * avg_steps + j]
    maddpg_fix_ep_len[i] = maddpg_fix_ep_len[i] / avg_steps
maddpg_att_fix_ep_len = np.zeros((int(np.size(maddpg_att_fix_ep_len_data) / avg_steps),))
for i in range(np.size(maddpg_att_fix_ep_len)):
    for j in range(avg_steps):
        maddpg_att_fix_ep_len[i] += maddpg_att_fix_ep_len_data[i * avg_steps + j]
    maddpg_att_fix_ep_len[i] = maddpg_att_fix_ep_len[i] / avg_steps
maddpg_lstm_att_fix_ep_len = np.zeros((int(np.size(maddpg_lstm_att_fix_ep_len_data) / avg_steps),))
for i in range(np.size(maddpg_lstm_att_fix_ep_len)):
    for j in range(avg_steps):
        maddpg_lstm_att_fix_ep_len[i] += maddpg_lstm_att_fix_ep_len_data[i * avg_steps + j]
    maddpg_lstm_att_fix_ep_len[i] = maddpg_lstm_att_fix_ep_len[i] / avg_steps

x_label1 = np.arange(0, np.size(ppo_fix_ep_len), 1)
x_label2 = np.arange(0, np.size(maddpg_fix_ep_len), 1)
x_label3 = np.arange(0, np.size(maddpg_att_fix_ep_len), 1)
x_label4 = np.arange(0, np.size(maddpg_lstm_att_fix_ep_len), 1)
plt.figure()
# plt.plot(x_label1 * 1e6 / np.size(x_label1), ppo_fix_ep_len, 'b', label='ppo')
plt.plot(x_label2 * 1e6 / np.size(x_label2), maddpg_fix_ep_len, 'b', label='maddpg')
plt.plot(x_label3 * 1e6 / np.size(x_label3), maddpg_att_fix_ep_len, 'g', label='maddpg_att')
plt.plot(x_label4 * 1e6 / np.size(x_label4), maddpg_lstm_att_fix_ep_len, 'r', label='maddpg_lstm_att')
plt.xlabel("time_steps")
plt.ylabel("ep_len_mean")
plt.title("fix_environment")
plt.legend()
# plt.show()

last_steps = int(4400)
maddpg_fix_ep_findnums_avg = sum(maddpg_fix_findnums_data[np.size(maddpg_fix_findnums_data) - last_steps:np.size(maddpg_fix_findnums_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(maddpg_fix_findnums_data) - last_steps, np.size(maddpg_fix_findnums_data)):
    sum1 += (maddpg_fix_findnums_data[i] - maddpg_fix_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
maddpg_fix_ep_rew_avg = sum(maddpg_fix_ep_rew_data[np.size(maddpg_fix_ep_rew_data) - last_steps:np.size(maddpg_fix_ep_rew_data)]) / last_steps
sum2 = 0.0
for i in range(np.size(maddpg_fix_ep_rew_data) - last_steps, np.size(maddpg_fix_ep_rew_data)):
    sum2 += (maddpg_fix_ep_rew_data[i] - maddpg_fix_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
maddpg_fix_ep_len_avg = sum(maddpg_fix_ep_len_data[np.size(maddpg_fix_ep_len_data) - last_steps:np.size(maddpg_fix_ep_len_data)]) / last_steps
sum3 = 0.0
for i in range(np.size(maddpg_fix_ep_len_data) - last_steps, np.size(maddpg_fix_ep_len_data)):
    sum3 += (maddpg_fix_ep_len_data[i] - maddpg_fix_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('maddpg_fix_ep_findnums_avg: {}, sigma: {}'.format(maddpg_fix_ep_findnums_avg, sigma1))
print('maddpg_fix_ep_rew_avg: {}, sigma: {}'.format(maddpg_fix_ep_rew_avg, sigma2))
print('maddpg_fix_ep_len_avg: {}, sigma: {}'.format(maddpg_fix_ep_len_avg, sigma3))

maddpg_att_fix_ep_findnums_avg = sum(maddpg_att_fix_findnums_data[np.size(maddpg_att_fix_findnums_data) - last_steps:np.size(maddpg_att_fix_findnums_data)]) / last_steps
maddpg_att_fix_ep_rew_avg = sum(maddpg_att_fix_ep_rew_data[np.size(maddpg_att_fix_ep_rew_data) - last_steps:np.size(maddpg_att_fix_ep_rew_data)]) / last_steps
maddpg_att_fix_ep_len_avg = sum(maddpg_att_fix_ep_len_data[np.size(maddpg_att_fix_ep_len_data) - last_steps:np.size(maddpg_att_fix_ep_len_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(maddpg_att_fix_findnums_data) - last_steps, np.size(maddpg_att_fix_findnums_data)):
    sum1 += (maddpg_att_fix_findnums_data[i] - maddpg_att_fix_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
maddpg_att_fix_ep_rew_avg = sum(maddpg_att_fix_ep_rew_data[np.size(maddpg_att_fix_ep_rew_data) - last_steps:np.size(maddpg_att_fix_ep_rew_data)]) / last_steps
sum2 = 0.0
for i in range(np.size(maddpg_att_fix_ep_rew_data) - last_steps, np.size(maddpg_att_fix_ep_rew_data)):
    sum2 += (maddpg_att_fix_ep_rew_data[i] - maddpg_att_fix_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
maddpg_att_fix_ep_len_avg = sum(maddpg_att_fix_ep_len_data[np.size(maddpg_att_fix_ep_len_data) - last_steps:np.size(maddpg_att_fix_ep_len_data)]) / last_steps
sum3 = 0.0
for i in range(np.size(maddpg_att_fix_ep_len_data) - last_steps, np.size(maddpg_att_fix_ep_len_data)):
    sum3 += (maddpg_att_fix_ep_len_data[i] - maddpg_att_fix_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('maddpg_att_fix_ep_findnums_avg: {}, sigma: {}'.format(maddpg_att_fix_ep_findnums_avg, sigma1))
print('maddpg_att_fix_ep_rew_avg: {}, sigma: {}'.format(maddpg_att_fix_ep_rew_avg, sigma2))
print('maddpg_att_fix_ep_len_avg: {}, sigma: {}'.format(maddpg_att_fix_ep_len_avg, sigma3))

maddpg_lstm_att_fix_ep_findnums_avg = sum(maddpg_lstm_att_fix_findnums_data[np.size(maddpg_lstm_att_fix_findnums_data) - last_steps:np.size(maddpg_lstm_att_fix_findnums_data)]) / last_steps
maddpg_lstm_att_fix_ep_rew_avg = sum(maddpg_lstm_att_fix_ep_rew_data[np.size(maddpg_lstm_att_fix_ep_rew_data) - last_steps:np.size(maddpg_lstm_att_fix_ep_rew_data)]) / last_steps
maddpg_lstm_att_fix_ep_len_avg = sum(maddpg_lstm_att_fix_ep_len_data[np.size(maddpg_lstm_att_fix_ep_len_data) - last_steps:np.size(maddpg_lstm_att_fix_ep_len_data)]) / last_steps
sum1 = 0.0
for i in range(np.size(maddpg_lstm_att_fix_findnums_data) - last_steps, np.size(maddpg_lstm_att_fix_findnums_data)):
    sum1 += (maddpg_lstm_att_fix_findnums_data[i] - maddpg_lstm_att_fix_ep_findnums_avg) ** 2
sum1 = sum1 / last_steps
sigma1 = np.sqrt(sum1)
maddpg_lstm_att_fix_ep_rew_avg = sum(maddpg_lstm_att_fix_ep_rew_data[np.size(maddpg_lstm_att_fix_ep_rew_data) - last_steps:np.size(maddpg_lstm_att_fix_ep_rew_data)]) / last_steps
sum2 = 0.0
for i in range(np.size(maddpg_lstm_att_fix_ep_rew_data) - last_steps, np.size(maddpg_lstm_att_fix_ep_rew_data)):
    sum2 += (maddpg_lstm_att_fix_ep_rew_data[i] - maddpg_lstm_att_fix_ep_rew_avg) ** 2
sum2 = sum2 / last_steps
sigma2 = np.sqrt(sum2)
maddpg_lstm_att_fix_ep_len_avg = sum(maddpg_lstm_att_fix_ep_len_data[np.size(maddpg_lstm_att_fix_ep_len_data) - last_steps:np.size(maddpg_lstm_att_fix_ep_len_data)]) / last_steps
sum3 = 0.0
for i in range(np.size(maddpg_lstm_att_fix_ep_len_data) - last_steps, np.size(maddpg_lstm_att_fix_ep_len_data)):
    sum3 += (maddpg_lstm_att_fix_ep_len_data[i] - maddpg_lstm_att_fix_ep_len_avg) ** 2
sum3 = sum3 / last_steps
sigma3 = np.sqrt(sum3)
print('maddpg_lstm_att_fix_ep_findnums_avg: {}, sigma: {}'.format(maddpg_lstm_att_fix_ep_findnums_avg, sigma1))
print('maddpg_lstm_att_fix_ep_rew_avg: {}, sigma: {}'.format(maddpg_lstm_att_fix_ep_rew_avg, sigma2))
print('maddpg_lstm_att_fix_ep_len_avg: {}, sigma: {}'.format(maddpg_lstm_att_fix_ep_len_avg, sigma3))

plt.show()













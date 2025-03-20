import csv
import matplotlib.pyplot as plt
import numpy as np

maddpg_data = np.loadtxt('./maddpg/maddpg_fix_4_30_rew.csv', delimiter=',')
maddpg_la_data = np.loadtxt('./maddpg/maddpg_la_4_30_rew.csv', delimiter=',')
# td3_data = np.loadtxt('./td3/td3_fix_4_rew_2.csv', delimiter=',')
# ddpg_data = np.loadtxt('./ddpg/ddpg_fix_4_rew_2.csv', delimiter=',')
# td3_la_data = np.loadtxt('./td3/td3_laf_4_rew_2.csv', delimiter=',')
# ddpg_la_data = np.loadtxt('./ddpg/ddpg_laf_4_rew_2.csv', delimiter=',')
# maddpg_data_standard = np.loadtxt('./maddpg_standard/maddpg_laf_rew.csv', delimiter=',')

avg_steps = 200
maddpg_rew = np.zeros((int(np.size(maddpg_data) / avg_steps),))
for i in range(np.size(maddpg_rew)):
    for j in range(avg_steps):
        maddpg_rew[i] += maddpg_data[i * avg_steps + j]
    maddpg_rew[i] = maddpg_rew[i] / avg_steps
maddpg_la_rew = np.zeros((int(np.size(maddpg_la_data) / avg_steps),))
for i in range(np.size(maddpg_la_rew)):
    for j in range(avg_steps):
        maddpg_la_rew[i] += maddpg_la_data[i * avg_steps + j]
    maddpg_la_rew[i] = maddpg_la_rew[i] / avg_steps
# maddpg_rew_standard = np.zeros((int(np.size(maddpg_data_standard) / avg_steps),))
# for i in range(np.size(maddpg_rew_standard)):
#     for j in range(avg_steps):
#         maddpg_rew_standard[i] += maddpg_data_standard[i * avg_steps + j]
#     maddpg_rew_standard[i] = maddpg_rew_standard[i] / avg_steps
# avg_steps = 5
# td3_rew = np.zeros((int(np.size(td3_data[:, 2]) / avg_steps),))
# for i in range(np.size(td3_rew)):
#     for j in range(avg_steps):
#         td3_rew[i] += td3_data[i * avg_steps + j, 2]
#     td3_rew[i] = td3_rew[i] / avg_steps
# td3_la_rew = np.zeros((int(np.size(td3_la_data[:, 2]) / avg_steps),))
# for i in range(np.size(td3_la_rew)):
#     for j in range(avg_steps):
#         td3_la_rew[i] += td3_la_data[i * avg_steps + j, 2]
#     td3_la_rew[i] = td3_la_rew[i] / avg_steps
# ddpg_rew = np.zeros((int(np.size(ddpg_data[:, 2]) / avg_steps),))
# for i in range(np.size(ddpg_rew)):
#     for j in range(avg_steps):
#         ddpg_rew[i] += ddpg_data[i * avg_steps + j, 2]
#     ddpg_rew[i] = ddpg_rew[i] / avg_steps
# ddpg_la_rew = np.zeros((int(np.size(ddpg_la_data[:, 2]) / avg_steps),))
# for i in range(np.size(ddpg_la_rew)):
#     for j in range(avg_steps):
#         ddpg_la_rew[i] += ddpg_la_data[i * avg_steps + j, 2]
#     ddpg_la_rew[i] = ddpg_la_rew[i] / avg_steps

x_label_1 = np.arange(0, np.size(maddpg_rew), 1)
x_label_2 = np.arange(0, np.size(maddpg_la_rew), 1)
# x_label_3 = np.arange(0, np.size(td3_rew), 1)
# x_label_4 = np.arange(0, np.size(td3_la_rew), 1)
# x_label_5 = np.arange(0, np.size(ddpg_rew), 1)
# x_label_6 = np.arange(0, np.size(ddpg_la_rew), 1)

plt.figure(1)
plt.plot(x_label_1, maddpg_rew, color=(1, 0, 0), label='maddpg')
plt.plot(x_label_2, maddpg_la_rew, color=(0, 0, 1), label='maddpg_lstm_att')
# plt.plot(x_label_3 * 5e6 / np.size(td3_rew), td3_rew, color='k', label='td3')
# plt.plot(x_label_4 * 5e6 / np.size(td3_la_rew), td3_la_rew, color='g', label='td3_lstm_att')
# plt.plot(x_label_5 * 5e6 / np.size(ddpg_rew), ddpg_rew, color=(0.5, 0, 0.5), label='ddpg')
# plt.plot(x_label_6 * 5e6 / np.size(ddpg_la_rew), ddpg_la_rew, color=(0.5, 0.5, 0.5), label='ddpg_lstm_att')
plt.xlabel("training steps")
plt.ylabel("average episode reward")
plt.legend()
plt.show()




























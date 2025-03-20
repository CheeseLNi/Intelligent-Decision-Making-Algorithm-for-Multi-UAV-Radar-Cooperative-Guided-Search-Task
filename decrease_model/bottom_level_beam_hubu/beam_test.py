import numpy as np
import matplotlib.pyplot as plt
import torch
from stable_baselines3 import TD3
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
import gym
from gym import spaces
from copy import deepcopy

def compute_omega(Pav, Gt, Gr, lamda, sigma, k, T0, Fn, L):
    omega = Pav * Gt * Gr * lamda ** 2 * sigma / ((4.0 * np.pi) ** 3 * k * T0 * Fn * L)
    return omega

def dB2W(dB):
    return 10.0 ** (dB / 10.0)

def compute_threat(n, type, v, R):
    T_type = np.zeros((n,))
    T_R = np.zeros((n,))
    T_v = np.zeros((n,))
    R_min = min(R) - (R[1] - R[0])
    R_max = max(R) + (R[1] - R[0])
    v_min = min(v) - (v[1] - v[0])
    v_max = max(v) + (v[1] - v[0])
    for i in range(n):
        T_type[i] = type[i]

        if R[i] <= R_min:
            T_R[i] = 1.0
        elif R[i] < R_max:
            T_R[i] = (R_max - R[i]) / (R_max - R_min)
        else:
            T_R[i] = 0.0

        if v[i] <= v_min:
            T_v[i] = 0.0
        elif v[i] < v_max:
            T_v[i] = (v[i] - v_min) / (v_max - v_min)
        else:
            T_v[i] = 1.0

    w_1 = T_type * T_v * T_R
    w = w_1 / sum(w_1)
    return w

# 需要搞一个固定贪心策略的数据集
def compute_airspace_target_cover(x_m, y_m, scan_range_x, scan_range_y):
    # 输入：x_m 目标方位角，y_m 目标俯仰角，scan_range_x 空域方位扫描范围，scan_range_y 目标俯仰扫描范围

    # 空域范围包含方位[-90, 90]，俯仰[-90, 90]
    # 方位空域数目=(180-scan_range_x)/5+1，俯仰空域数目=(180-scan_range_x)/5+1
    airspace_num_x = int((180 - scan_range_x) / 5 + 1)
    airspace_num_y = int((180 - scan_range_y) / 5 + 1)
    target_num = np.size(x_m)
    airspace_cover_target = np.zeros((airspace_num_x * airspace_num_y, target_num))  # 整个空域的覆盖情况
    airspace_cover_num = np.zeros((airspace_num_x * airspace_num_y,))  # 每个空域包含的目标数目
    target_covered_airspace = np.zeros((target_num, int(scan_range_x/5) * int(scan_range_y/5)))   # 每个目标被哪些子空域包含
    target_covered_num = np.zeros((target_num,))    # 被包含子空域的数量
    for i in range(airspace_num_x):
        for j in range(airspace_num_y):
            airspace_center_x = -90 + scan_range_x / 2 + i * 5.0
            airspace_center_y = 90 - scan_range_y / 2 - j * 5.0
            for k in range(target_num):
                if (abs(x_m[k] - airspace_center_x) < scan_range_x / 2) and (abs(y_m[k] - airspace_center_y) < scan_range_y / 2):
                    airspace_cover_target[i + 0 + airspace_num_x * j, int(airspace_cover_num[i + 0 + airspace_num_x * j])] = k + 1
                    airspace_cover_num[i + 0 + airspace_num_x * j] += 1
                    target_covered_airspace[k, int(target_covered_num[k])] = i + 0 + airspace_num_x * j + 1     # 对应空域编号
                    target_covered_num[k] += 1

    return airspace_cover_target, airspace_cover_num, target_covered_airspace, target_covered_num

def compute_airspace_cover_target_(airspace_cover_target, airspace_cover_num, airspace_bh):
    airspace_cover_target_ = []
    for i in range(int(airspace_cover_num[airspace_bh])):
        target_bh = airspace_cover_target[airspace_bh, i]
        airspace_cover_target_.append(target_bh)
    return airspace_cover_target_

def compute_airspace_center_xy(x_m, y_m, scan_range_x, scan_range_y, target_cover_situation):
    global_target_uncover = []
    target_num = np.size(x_m)
    airspace_cover_target, airspace_cover_num, target_covered_airspace, target_covered_num = compute_airspace_target_cover(x_m, y_m, scan_range_x, scan_range_y)
    # 求出未被覆盖的目标集合
    for i in range(target_num):
        if target_cover_situation[i] == 0:  # 说明目标未被覆盖
            target_bh = int(i + 1)
            global_target_uncover.append(target_bh)
    # 选集合的第一个目标作为本次搜索的对象
    target_bh = int(global_target_uncover[0] - 1)   # 目标编号
    target_covered_num_ = int(target_covered_num[target_bh])    # 计算有多少个空域包含该目标
    cover_num = np.zeros((target_covered_num_,))  # 包含该目标的子空域包含未发现目标的数量
    for j in range(target_covered_num_):
        airspace_bh = int(target_covered_airspace[target_bh, j] - 1)    # 空域编号
        airspace_cover_target_ = compute_airspace_cover_target_(airspace_cover_target, airspace_cover_num, airspace_bh)  # 计算空域包含的目标集合
        cover = [x for x in airspace_cover_target_ if x in global_target_uncover]   # 计算空域包含的未发现目标的集合
        cover_num[j] = np.size(cover)
    cover_num = list(cover_num)     # 得到所有包含该目标的子空域各自包含未发现目标的数量
    cover_num_max_index = cover_num.index(max(cover_num))
    '''寻找包含该目标的子空域中包含未发现目标数量最多的空域'''
    airspace_good_cover = []    # 包含该目标的子空域中包含未发现目标数量最多的空域，肯定有很多空域包含的数目是一样的
    for i in range(target_covered_num_):
        if abs(cover_num[i] - max(cover_num)) < 1e-6:
            airspace_bh = int(target_covered_airspace[target_bh, i] - 1)
            airspace_good_cover.append(airspace_bh)
    '''在未发现目标数量最多的空域中，找到目标最靠近空域中心的那个'''
    distance_average = np.zeros((np.size(airspace_good_cover),))
    for i in range(np.size(airspace_good_cover)):
        airspace_bh = int(airspace_good_cover[i])
        airspace_cover_num_ = int(airspace_cover_num[airspace_bh])  # 该空域包含目标的数量
        airspace_num_x = int((180 - scan_range_x) / 5 + 1)
        hang = np.fix(airspace_bh / airspace_num_x)  # 计算空域在第几行
        lie = airspace_bh - hang * airspace_num_x  # 计算空域在第几列
        airspace_center_x = -90 + scan_range_x / 2 + lie * 5.0
        airspace_center_y = 90 - scan_range_y / 2 - hang * 5.0
        for j in range(airspace_cover_num_):
            target_bh = int(airspace_cover_target[airspace_bh, j] - 1)
            distance = np.sqrt((x_m[target_bh] - airspace_center_x)**2 + (y_m[target_bh] - airspace_center_y)**2)
            distance_average[i] += distance
        distance_average[i] = distance_average[i] / airspace_cover_num_
    distance_average = list(distance_average)
    distance_shortest_index = distance_average.index(min(distance_average))
    airspace_best_bh = int(airspace_good_cover[distance_shortest_index])

    """计算最优空域中心"""
    airspace_num_x = int((180 - scan_range_x) / 5 + 1)
    hang = np.fix(airspace_best_bh / airspace_num_x)     # 计算空域在第几行
    lie = airspace_best_bh - hang * airspace_num_x   # 计算空域在第几列
    airspace_center_x = -90 + scan_range_x / 2 + lie * 5.0
    airspace_center_y = 90 - scan_range_y / 2 - hang * 5.0
    # 更新目标覆盖情况
    target_cover_situation_new = deepcopy(target_cover_situation)
    airspace_cover_target_ = compute_airspace_cover_target_(airspace_cover_target, airspace_cover_num, airspace_best_bh)
    airspace_cover_num_ = int(airspace_cover_num[airspace_best_bh])
    for i in range(airspace_cover_num_):
        target_bh = int(airspace_cover_target[airspace_best_bh, i] - 1)
        target_cover_situation_new[target_bh] += 1

    return airspace_center_x, airspace_center_y, target_cover_situation_new

def compute_betaij(x_m, y_m, threat_w, Mi, scan_center, scan_range):
    tgt_num = np.size(x_m)
    scan_center_x, scan_center_y = scan_center[0], scan_center[1]
    scan_range_x, scan_range_y = scan_range[0], scan_range[1]

    """计算各子空域中心"""
    sub_center = np.zeros((Mi, 2))
    sub_range = np.zeros((Mi, 2))
    if Mi == 1:
        sub_center[0, 0] = scan_center_x
        sub_center[0, 1] = scan_center_y
        sub_range[0, 0] = scan_range_x
        sub_range[0, 1] = scan_range_y
    if Mi == 4:
        sub_center[0, 0] = scan_center_x - scan_range_x / 4
        sub_center[1, 0] = scan_center_x + scan_range_x / 4
        sub_center[2, 0] = sub_center[0, 0]
        sub_center[3, 0] = sub_center[1, 0]

        sub_center[0, 1] = scan_center_y + scan_range_x / 4
        sub_center[1, 1] = sub_center[0, 1]
        sub_center[2, 1] = scan_center_y - scan_range_x / 4
        sub_center[3, 1] = sub_center[2, 1]

        sub_range[:, 0] = scan_range_x / 2
        sub_range[:, 1] = scan_range_y / 2

    """计算每个子空域包含目标情况"""
    zikongyu = []
    for i in range(Mi):
        zikongyu_ = []  # 暂时存放
        for j in range(tgt_num):
            if abs(x_m[j] - sub_center[i, 0]) < sub_range[i, 0] / 2 and abs(y_m[j] - sub_center[i, 1]) < sub_range[i, 1] / 2:
                zikongyu_.append(int(j + 1))
        zikongyu.append(zikongyu_)
    """计算每个子空域的betaij"""
    betaij = np.zeros((Mi,))
    for i in range(Mi):
        for j in range(np.size(zikongyu[i])):
            target_bh = int(zikongyu[i][j] - 1)
            betaij[i] += threat_w[target_bh]  # 计算每个空域威胁度
        if betaij[i] < 1e-6:
            betaij[i] = 0.01
    betaij = betaij / sum(betaij)

    tgt_num_ij = []
    for i in range(Mi):
        tgt_num_ij.append(np.size(zikongyu[i]))

    return betaij, tgt_num_ij

def plot_airspace(center_x, center_y, range_x, range_y, color, linestyle):
    plt.plot([center_x - range_x / 2, center_x + range_x / 2],
             [center_y - range_y / 2, center_y - range_y / 2], color=color, linestyle=linestyle)
    plt.plot([center_x + range_x / 2, center_x + range_x / 2],
             [center_y - range_y / 2, center_y + range_y / 2], color=color, linestyle=linestyle)
    plt.plot([center_x + range_x / 2, center_x - range_x / 2],
             [center_y + range_y / 2, center_y + range_y / 2], color=color, linestyle=linestyle)
    plt.plot([center_x - range_x / 2, center_x - range_x / 2],
             [center_y + range_y / 2, center_y - range_y / 2], color=color, linestyle=linestyle)

def plot_sub_airspace(Mi, scan_center, scan_range, color):
    scan_center_x, scan_center_y = scan_center[0], scan_center[1]
    scan_range_x, scan_range_y = scan_range[0], scan_range[1]
    """计算各子空域中心"""
    sub_center = np.zeros((Mi, 2))
    sub_range = np.zeros((Mi, 2))
    if Mi == 4:
        plt.plot([scan_center_x - scan_range_x / 2, scan_center_x + scan_range_x / 2], [scan_center_y, scan_center_y], color=color, linestyle='--')
        plt.plot([scan_center_x, scan_center_x], [scan_center_y - scan_range_y / 2, scan_center_y + scan_range_x / 2], color=color, linestyle='--')

def track_marker(x, y, z, color):
    r = 2.0
    theta = np.linspace(0.0, np.pi * 2, 100)
    _x = r * np.cos(theta) + x * np.ones((100,))
    _y = r * np.sin(theta) + y * np.ones((100,))
    _z = 0.0 + z * np.ones((100,))
    plt.plot(_x, _y, _z, color)

    theta = np.linspace(0.0, np.pi * 2, 100)
    _x = r * np.sin(theta) + x * np.ones((100,))
    _y = 0.0 + y * np.ones((100,))
    _z = r * np.cos(theta) + z * np.ones((100,))
    plt.plot(_x, _y, _z, color)

    plt.pause(0.01)

def track_marker_2d(x, y, color):
    r = 2.0
    theta = np.linspace(0.0, np.pi * 2, 100)
    _x = r * np.cos(theta) + x * np.ones((100,))
    _y = r * np.sin(theta) + y * np.ones((100,))
    plt.plot(_x, _y, color)
    plt.pause(0.01)

def display_xietong_beam(center_x, center_y, beamwidth, Rd, colors, xietong_num):
    lines = []
    for i in range(xietong_num):
        # 求四个边角点
        azimuth = (center_x[i] + beamwidth[i] / 2.0) * np.pi / 180.0
        pitch = (center_y[i] + beamwidth[i] / 2.0) * np.pi / 180.0
        xt_1 = Rd * np.cos(pitch) * np.cos(azimuth)
        yt_1 = Rd * np.cos(pitch) * np.sin(azimuth)
        zt_1 = Rd * np.sin(pitch)

        azimuth = (center_x[i] - beamwidth[i] / 2.0) * np.pi / 180.0
        pitch = (center_y[i] + beamwidth[i] / 2.0) * np.pi / 180.0
        xt_2 = Rd * np.cos(pitch) * np.cos(azimuth)
        yt_2 = Rd * np.cos(pitch) * np.sin(azimuth)
        zt_2 = Rd * np.sin(pitch)

        azimuth = (center_x[i] - beamwidth[i] / 2.0) * np.pi / 180.0
        pitch = (center_y[i] - beamwidth[i] / 2.0) * np.pi / 180.0
        xt_3 = Rd * np.cos(pitch) * np.cos(azimuth)
        yt_3 = Rd * np.cos(pitch) * np.sin(azimuth)
        zt_3 = Rd * np.sin(pitch)

        azimuth = (center_x[i] + beamwidth[i] / 2.0) * np.pi / 180.0
        pitch = (center_y[i] - beamwidth[i] / 2.0) * np.pi / 180.0
        xt_4 = Rd * np.cos(pitch) * np.cos(azimuth)
        yt_4 = Rd * np.cos(pitch) * np.sin(azimuth)
        zt_4 = Rd * np.sin(pitch)

        line1 = plt.plot([xt_1, xt_2], [yt_1, yt_2], [zt_1, zt_2], color=colors[i], linestyle='-')
        line2 = plt.plot([xt_2, xt_3], [yt_2, yt_3], [zt_2, zt_3], color=colors[i], linestyle='-')
        line3 = plt.plot([xt_3, xt_4], [yt_3, yt_4], [zt_3, zt_4], color=colors[i], linestyle='-')
        line4 = plt.plot([xt_4, xt_1], [yt_4, yt_1], [zt_4, zt_1], color=colors[i], linestyle='-')

        line5 = plt.plot([xt_1, 0.0], [yt_1, 0.0], [zt_1, 0.0], color=colors[i], linestyle='--')
        line6 = plt.plot([xt_2, 0.0], [yt_2, 0.0], [zt_2, 0.0], color=colors[i], linestyle='--')
        line7 = plt.plot([xt_3, 0.0], [yt_3, 0.0], [zt_3, 0.0], color=colors[i], linestyle='--')
        line8 = plt.plot([xt_4, 0.0], [yt_4, 0.0], [zt_4, 0.0], color=colors[i], linestyle='--')

        # lines.append([line5[0], line6[0], line7[0], line8[0]])
        lines.append([line1[0], line2[0], line3[0], line4[0], line5[0], line6[0], line7[0], line8[0]])
    plt.pause(0.01)
    for i in range(xietong_num):
        for line in lines[i]:
            line.remove()

def display_xietong_beam_2d(center_x, center_y, beamwidth, colors, xietong_num):
    lines = []
    for i in range(xietong_num):
        x_min = center_x[i] - beamwidth[i] / 2.0
        x_max = center_x[i] + beamwidth[i] / 2.0
        y_min = center_y[i] - beamwidth[i] / 2.0
        y_max = center_y[i] + beamwidth[i] / 2.0

        line1 = plt.plot([x_max, x_max], [y_min, y_max], color=colors[i], linestyle='--')
        line2 = plt.plot([x_max, x_min], [y_max, y_max], color=colors[i], linestyle='--')
        line3 = plt.plot([x_min, x_min], [y_max, y_min], color=colors[i], linestyle='--')
        line4 = plt.plot([x_min, x_max], [y_min, y_min], color=colors[i], linestyle='--')

        lines.append([line1[0], line2[0], line3[0], line4[0]])
    plt.pause(0.01)
    for i in range(xietong_num):
        for line in lines[i]:
            line.remove()


glo_tgt_num = 30  # 目标总数目
glo_co_num = 3   # 协同雷达数目
sub_as_num = [1, 4, 4]  # 各雷达搜索子空域划分
glo_scan_range_x = [10, 20, 30]  # 方位搜索范围
glo_scan_range_y = [10, 20, 30]  # 俯仰搜索范围

# 雷达系统参数
lamda = 9.0e-2  # S波段相控阵雷达波长，单位m
Pt = 6.0e5  # 发射功率，单位W
zhankongbi = 0.05  # 最高占空比为5%
Gt = 40.0  # 阵面法向处发射增益，单位dB
Gr = 40.0  # 接收天线增益，单位dB
RCS_sigma = 1  # 目标RCS，单位m^2
Fn = 3  # 接收机噪声系数，单位dB
L = 5  # 系统损耗，单位dB
T0 = 290  # 接收机噪声温度，常温下为290K
Pav = Pt * zhankongbi  # 平均功率，单位W
k = 1.380649e-23  # 玻尔兹曼常数，单位J/K

# 载机雷达参数设置
N = 3   # 载机雷达数
alpha_i = [0.2, 0.3, 0.5]    # 载机雷达权值
pd = [0.85, 0.9, 0.95]  # 检测概率
pfa = [2.0e-6, 1.5e-6, 1.0e-6]  # 虚警率
SNR_D = []
for i in range(N):
    SNR_D.append(np.log(pfa[i]) / np.log(pd[i]) - 1.0)
_omega = compute_omega(Pav, dB2W(Gt), dB2W(Gr), lamda, RCS_sigma, k, T0, dB2W(Fn), dB2W(L))
omega = [0.1 * _omega / 1000.0, _omega / 1000.0, 10.0 * _omega / 1000.0]    # 雷达系统常数

Mi = [1, 4, 4]  # 载机雷达对应子空域数
beamwidth = [2.0, 2.5, 3.0]     # 波束宽度
scan_range_x, scan_range_y = [10.0, 20.0, 30.0], [10.0, 20.0, 30.0]     # 载机雷达搜索范围
bowei_num_x = []    # 方位波位数
bowei_num_y = []    # 俯仰波位数
Nsi = []     # 搜索空域波位数
for i in range(N):
    bowei_num_x.append(int(scan_range_x[i] / beamwidth[i]))
    bowei_num_y.append(int(scan_range_y[i] / beamwidth[i]))
    Nsi.append(int(bowei_num_x[i] * bowei_num_y[i]))
print(Nsi)  # 各子空域波位数：25，16，25

# 目标参数设置，这些威胁度全部是固定的，只有目标坐标是随机的
target_num = 30
mb_type = np.arange(0.8, 1.0, 0.2 / target_num)  # 目标类型威胁度
v = np.arange(900.0, 1200.0, 300.0 / target_num)  # 目标速度，单位m/s
R = np.arange(200.0, 400.0, 200.0 / target_num)  # 目标距离，单位km
threat_w = compute_threat(target_num, mb_type, v, R)  # 每个目标的威胁度
sum_w_v = sum(threat_w * v)  # 计算平均速度

colors = ['red', 'green', 'blue']

x_m = np.random.uniform(-60.0, 60.0, size=(target_num,))
y_m = np.random.uniform(-60.0, 60.0, size=(target_num,))
# plt.figure(1)
# plt.plot(self.x_m, self.y_m, 'k^')
# plt.axis('equal')
# 随机初始化目标发现状况
target_cover_situation = np.zeros((target_num,))  # 空域覆盖状态，0，1对应位置目标是否被覆盖
target_cover_situation_flag = np.random.uniform(0.0, 1.0, size=(target_num,))
for i in range(target_num):
    if target_cover_situation_flag[i] <= 0.5:
        target_cover_situation[i] = 1
# 进行一次空域划分，顺便计算betaij
betaij = []  # 首先计算各子空域betaij
tgt_numij = []
cover_num = []
airspace_center_x_record, airspace_center_y_record = [], []
plt.figure(1)
for i in range(N):
    airspace_center_x, airspace_center_y, target_cover_situation = \
        compute_airspace_center_xy(x_m, y_m, scan_range_x[i], scan_range_y[i], target_cover_situation)
    airspace_center_x_record.append(airspace_center_x)
    airspace_center_y_record.append(airspace_center_y)
    _cover_num = 0
    for j in range(target_num):
        if abs(airspace_center_x - x_m[j]) < scan_range_x[i] / 2.0 and abs(airspace_center_y - y_m[j]) < scan_range_y[i] / 2.0:
            _cover_num += 1
    _betaij, _tgy_numij = compute_betaij(x_m, y_m, threat_w, Mi[i], [airspace_center_x, airspace_center_y], [scan_range_x[i], scan_range_y[i]])
    # plot_airspace(scan_range_x[i], scan_range_y[i], airspace_center_x, airspace_center_y, colors[i])
    # plot_sub_airspace(Mi[i], [airspace_center_x, airspace_center_y], [scan_range_x[i], scan_range_y[i]], colors[i])
    cover_num.append(_cover_num)
    betaij.append(_betaij)
    tgt_numij.append(_tgy_numij)
print(cover_num)
print(betaij)
print(tgt_numij)
#######################################################计算srij##########################################################
srij = []
for i in range(N):
    srj = np.zeros((Mi[i],))
    for j in range(Mi[i]):
        Nsij = int(Nsi[i] / Mi[i])
        srj[j] = 1.0 / (betaij[i][j] ** (-4.0 / 3.0) * (Nsij / omega[i]) ** (1.0 / 3.0))
    srj = srj / sum(srj)
    srij.append(srj)
########################################################计算帧周期########################################################
tfi = np.zeros((N,))
for i in range(N):
    _tfi = 0
    for j in range(Mi[i]):
        Nsij = int(Nsi[i] / Mi[i])
        _tfi += betaij[i][j] * (omega[i] * srij[i][j] / (SNR_D[i] * Nsij)) ** (1.0 / 4.0)
    tfi[i] = (2 * sum_w_v / _tfi) ** (-4.0 / 3.0)
print(tfi)
#####################################################计算波束驻留时间######################################################
tao_sij = []
for i in range(N):
    tao_sj = []
    for j in range(Mi[i]):
        Nsij = int(Nsi[i] / Mi[i])
        tao_sj.append(srij[i][j] / Nsij * tfi[i])
    tao_sij.append(tao_sj)
print(tao_sij)
###################################################首先是正常搜耗时########################################################
kongyu_num = N
Rm = np.random.uniform(60.0, 80.0, (target_num,))
Rd = 80.0   # 雷达探测距离
x_m_3d = Rm * np.cos(y_m * np.pi / 180.0) * np.cos(x_m * np.pi / 180.0)
y_m_3d = Rm * np.cos(y_m * np.pi / 180.0) * np.sin(x_m * np.pi / 180.0)
z_m_3d = Rm * np.sin(y_m * np.pi / 180.0)
plt.ion()
plt.figure(1)
plt.subplot(projection='3d')
plt.plot(x_m_3d, y_m_3d, z_m_3d, 'k^', label='target')
plt.axis('equal')
# plt.show()

plt.figure(2)
plt.plot(x_m, y_m, 'k^', label='target')
plt.axis('equal')
# plt.show()

###################################################二维空域可视化######################################################
init_scan_range = []    # 另一个局部变量
plt.figure(2)
plt.axis('equal')
for i in range(kongyu_num):
    scan_x_min = airspace_center_x_record[i] - scan_range_x[i] / 2.0
    scan_x_max = airspace_center_x_record[i] + scan_range_x[i] / 2.0
    scan_y_min = airspace_center_y_record[i] - scan_range_y[i] / 2.0
    scan_y_max = airspace_center_y_record[i] + scan_range_y[i] / 2.0
    init_scan_range.append([[scan_x_min, scan_x_max], [scan_y_min, scan_y_max]])
    plot_airspace(airspace_center_x_record[i], airspace_center_y_record[i], scan_range_x[i], scan_range_y[i], colors[i], '-')
# plt.show()

###################################################三维空域可视化######################################################
angle_point = []  # 空域边角三维坐标
plt.figure(1)
for i in range(kongyu_num):
    _scan_range_x, _scan_range_y = init_scan_range[i][0], init_scan_range[i][1]
    scan_range_x_min, scan_range_x_max = _scan_range_x[0] * np.pi / 180.0, _scan_range_x[1] * np.pi / 180.0
    scan_range_y_min, scan_range_y_max = _scan_range_y[0] * np.pi / 180.0, _scan_range_y[1] * np.pi / 180.0

    xt_1 = Rd * np.cos(scan_range_y_max) * np.cos(scan_range_x_max)
    yt_1 = Rd * np.cos(scan_range_y_max) * np.sin(scan_range_x_max)
    zt_1 = Rd * np.sin(scan_range_y_max)

    xt_2 = Rd * np.cos(scan_range_y_max) * np.cos(scan_range_x_min)
    yt_2 = Rd * np.cos(scan_range_y_max) * np.sin(scan_range_x_min)
    zt_2 = Rd * np.sin(scan_range_y_max)

    xt_3 = Rd * np.cos(scan_range_y_min) * np.cos(scan_range_x_min)
    yt_3 = Rd * np.cos(scan_range_y_min) * np.sin(scan_range_x_min)
    zt_3 = Rd * np.sin(scan_range_y_min)

    xt_4 = Rd * np.cos(scan_range_y_min) * np.cos(scan_range_x_max)
    yt_4 = Rd * np.cos(scan_range_y_min) * np.sin(scan_range_x_max)
    zt_4 = Rd * np.sin(scan_range_y_min)

    angle_point.append([[xt_1, yt_1, zt_1], [xt_2, yt_2, zt_2], [xt_3, yt_3, zt_3], [xt_4, yt_4, zt_4]])

    plt.plot([xt_1, xt_2], [yt_1, yt_2], [zt_1, zt_2], color='{}'.format(colors[i]), linestyle='-')
    plt.plot([xt_2, xt_3], [yt_2, yt_3], [zt_2, zt_3], color='{}'.format(colors[i]), linestyle='-')
    plt.plot([xt_3, xt_4], [yt_3, yt_4], [zt_3, zt_4], color='{}'.format(colors[i]), linestyle='-')
    plt.plot([xt_4, xt_1], [yt_4, yt_1], [zt_4, zt_1], color='{}'.format(colors[i]), linestyle='-')
    plt.plot([xt_1, 0.0], [yt_1, 0.0], [zt_1, 0.0], color='{}'.format(colors[i]), linestyle='--')
    plt.plot([xt_2, 0.0], [yt_2, 0.0], [zt_2, 0.0], color='{}'.format(colors[i]), linestyle='--')
    plt.plot([xt_3, 0.0], [yt_3, 0.0], [zt_3, 0.0], color='{}'.format(colors[i]), linestyle='--')
    plt.plot([xt_4, 0.0], [yt_4, 0.0], [zt_4, 0.0], color='{}'.format(colors[i]), linestyle='--')
# plt.show()

#################################################计算各波位中心极坐标###################################################
beam_num_x, beam_num_y = [], []
beam_bh, beam_position = [], []
beam_tao_sij = []   # 各波位驻留时间
for i in range(kongyu_num):
    beam_num_x.append(int(scan_range_x[i] / beamwidth[i]))
    beam_num_y.append(int(scan_range_y[i] / beamwidth[i]))
    beam_bh.append(np.zeros((beam_num_x[i] * beam_num_y[i],)))
    beam_position.append(np.zeros((beam_num_x[i] * beam_num_y[i], 2)))
    beam_tao_sij.append(np.zeros((beam_num_x[i] * beam_num_y[i],)))
    init_scan_x = init_scan_range[i][0][0] + beamwidth[i] / 2.0
    init_scan_y = init_scan_range[i][1][1] - beamwidth[i] / 2.0
    for j in range(beam_num_x[i] * beam_num_y[i]):
        beam_bh[i][j] = j
        hang = int(j / beam_num_x[i])
        lie = j - hang * beam_num_x[i]
        beam_position[i][j][0] = init_scan_x + lie * beamwidth[i]
        beam_position[i][j][1] = init_scan_y - hang * beamwidth[i]
        if i == 0:
            beam_tao_sij[i][j] = tao_sij[0][0]
        else:
            if hang <= beam_num_y[i] / 2 and lie <= beam_num_y[i] / 2:
                beam_tao_sij[i][j] = tao_sij[i][0]
            elif hang <= beam_num_y[i] / 2 and lie > beam_num_y[i] / 2:
                beam_tao_sij[i][j] = tao_sij[i][1]
            elif hang > beam_num_y[i] / 2 and lie <= beam_num_y[i] / 2:
                beam_tao_sij[i][j] = tao_sij[i][2]
            else:
                beam_tao_sij[i][j] = tao_sij[i][3]

#################################################波束可视化时序搜索#####################################################
# 搜的过程可以分为两种，一种是搜自己的，一种是搜别人的
search_airpace_id = np.array([0, 1, 2])  # 当前搜索的空域id
search_beam_id = np.zeros((kongyu_num,))  # 当前搜索的波位id
airspace_search_num = np.ones((kongyu_num,))  # 每个空域的搜索雷达数量
airspace_search_id = [[0], [1], [2]]  # 每个空域的搜索雷达id
done = np.zeros((kongyu_num,))  # 任务完成情况
track_target_id = [[], [], []]  # 跟踪目标编号情况
track_target_airspace = [[], [], []]  # 跟踪目标空域情况
track_target_beam = [[], [], []]  # 跟踪目标分布波位情况
track_target_time = [[], [], []]  # 第一次发现目标的时间
real_time = np.zeros((kongyu_num,))
time = 0
# 波位时序搜索循环，这个循环需要改一下了，因为每个波位驻留时间不一样
while sum(done) < kongyu_num:
    xietong_search_x, xietong_search_y = [], []
    for i in range(kongyu_num):
        real_time[i] += beam_tao_sij[int(search_airpace_id[i])][int(search_beam_id[i])]
        init_scan_x = init_scan_range[int(search_airpace_id[i])][0][0] + beamwidth[i] / 2.0
        init_scan_y = init_scan_range[int(search_airpace_id[i])][1][1] - beamwidth[i] / 2.0
        scan_x = beam_position[int(search_airpace_id[i])][int(search_beam_id[i])][0]
        scan_y = beam_position[int(search_airpace_id[i])][int(search_beam_id[i])][1]  # 正常的搜索方位
        if track_target_id[i] == []:  # 未发现目标状态
            search_beam_id[i] += 1
            for j in range(target_num):
                if abs(x_m[j] - scan_x) < beamwidth[i] / 2.0 and abs(y_m[j] - scan_y) < beamwidth[i] / 2.0:
                    track_target_id[i].append(j)
                    track_target_airspace[i].append(search_airpace_id[i])
                    track_target_beam[i].append(search_beam_id[i])
                    track_target_time[i].append(time)
                    plt.figure(1)
                    track_marker(x_m_3d[j], y_m_3d[j], z_m_3d[j], colors[i])
                    plt.figure(2)
                    track_marker_2d(x_m[j], y_m[j], colors[i])
        else:  # 已开始跟踪目标，需要判断是否对目标进行跟踪
            track = 0
            for j in range(len(track_target_id[i])):
                if (time - track_target_time[i][j]) % 10 == 0 and time - track_target_time[i][j] > 0.0:
                    init_scan_x = init_scan_range[int(track_target_airspace[i][j])][0][0] + beamwidth[i] / 2.0
                    init_scan_y = init_scan_range[int(track_target_airspace[i][j])][1][1] - beamwidth[i] / 2.0
                    scan_x = beam_position[int(track_target_airspace[i][j])][int(track_target_beam[i][j])][0]
                    scan_y = beam_position[int(track_target_airspace[i][j])][int(track_target_beam[i][j])][1]  # 跟踪的搜索方位
                    track = 1
                    break
            if track == 0:
                _search_beam_id = []
                for k in range(int(airspace_search_num[int(search_airpace_id[i])])):
                    radar_id = airspace_search_id[int(search_airpace_id[i])][k]
                    _search_beam_id.append(search_beam_id[radar_id])
                search_beam_id[i] = max(_search_beam_id) + 1
                if search_beam_id[i] >= beam_num_x[int(search_airpace_id[i])] * beam_num_y[int(search_airpace_id[i])] - 1:
                    search_beam_id[i] = beam_num_x[int(search_airpace_id[i])] * beam_num_y[int(search_airpace_id[i])] - 1
                scan_x = beam_position[int(search_airpace_id[i])][int(search_beam_id[i])][0]
                scan_y = beam_position[int(search_airpace_id[i])][int(search_beam_id[i])][1]  #
                for j in range(target_num):
                    if abs(x_m[j] - scan_x) < beamwidth[i] / 2.0 and abs(y_m[j] - scan_y) < beamwidth[i] / 2.0:
                        track_target_id[i].append(j)
                        track_target_airspace[i].append(search_airpace_id[i])
                        track_target_beam[i].append(search_beam_id[i])
                        track_target_time[i].append(time)
                        plt.figure(1)
                        track_marker(x_m_3d[j], y_m_3d[j], z_m_3d[j], colors[i])
                        plt.figure(2)
                        track_marker_2d(x_m[j], y_m[j], colors[i])
        if search_beam_id[i] >= beam_num_x[int(search_airpace_id[i])] * beam_num_y[int(search_airpace_id[i])] - 1:
            done[int(search_airpace_id[i])] = 1
            for j in range(kongyu_num):
                if done[j] == 0:
                    airspace_search_num[int(search_airpace_id[i])] -= 1  # 原本搜索空域雷达数减一
                    airspace_search_id[int(search_airpace_id[i])].remove(i)
                    search_airpace_id[i] = j  # 搜索空域id更新
                    _search_beam_id = []
                    for k in range(int(airspace_search_num[int(search_airpace_id[i])])):
                        radar_id = airspace_search_id[j][k]
                        _search_beam_id.append(search_beam_id[radar_id])
                    airspace_search_num[int(search_airpace_id[i])] += 1  # 现在搜索空域雷达数加一
                    search_beam_id[i] = max(_search_beam_id) + 1  # 搜索波位更新
                    airspace_search_id[int(search_airpace_id[i])].append(i)  # 空域雷达跟新
                    break
        xietong_search_x.append(scan_x)
        xietong_search_y.append(scan_y)
        if sum(done) == kongyu_num:
            break
    time += 1
    if sum(done) < kongyu_num:
        plt.figure(1)
        display_xietong_beam(xietong_search_x, xietong_search_y, beamwidth, Rd, colors, kongyu_num)
        plt.figure(2)
        display_xietong_beam_2d(xietong_search_x, xietong_search_y, beamwidth, colors, kongyu_num)

print(real_time)

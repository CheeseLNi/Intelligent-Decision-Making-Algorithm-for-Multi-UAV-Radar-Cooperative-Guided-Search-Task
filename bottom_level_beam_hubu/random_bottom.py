import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy

beilv = 1.0
nijk_limit = [0.1, 0.1, 0.03]

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

def compute_uncover_situation(x_m, y_m, scan_range_x, scan_range_y, target_cover_situation):
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
                if (abs(x_m[k] - airspace_center_x) < scan_range_x / 2) and \
                        (abs(y_m[k] - airspace_center_y) < scan_range_y / 2) and target_cover_situation[k] == 0:
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

# 修改贪心算法，每次选取包含目标数目最多的一个空域
def greedy_algorithm(x_m, y_m, scan_range_x, scan_range_y, target_cover_situation):
    tgt_num = np.size(x_m)
    # 求出各空域包含未发现目标集合
    airspace_cover_target, airspace_cover_num, target_covered_airspace, target_covered_num = \
        compute_uncover_situation(x_m, y_m, scan_range_x, scan_range_y, target_cover_situation)
    airspace_num = np.size(airspace_cover_num)
    max_tgt_num_airspace_id = []  # 计算包含为发现目标数目最多的空域编号
    for i in range(airspace_num):
        if airspace_cover_num[i] == max(airspace_cover_num):
            max_tgt_num_airspace_id.append(i)
    # 在包含未发现目标最多的空域内部选取距离最小的空域
    distance_average = np.zeros((np.size(max_tgt_num_airspace_id),))
    for i in range(np.size(max_tgt_num_airspace_id)):
        airspace_bh = int(max_tgt_num_airspace_id[i])
        airspace_cover_num_ = int(airspace_cover_num[airspace_bh])  # 该空域包含目标的数量
        airspace_num_x = int((180 - scan_range_x) / 5 + 1)
        hang = np.fix(airspace_bh / airspace_num_x)  # 计算空域在第几行
        lie = airspace_bh - hang * airspace_num_x  # 计算空域在第几列
        airspace_center_x = -90 + scan_range_x / 2 + lie * 5.0
        airspace_center_y = 90 - scan_range_y / 2 - hang * 5.0
        for j in range(airspace_cover_num_):
            target_bh = int(airspace_cover_target[airspace_bh, j] - 1)
            distance = np.sqrt((x_m[target_bh] - airspace_center_x) ** 2 + (y_m[target_bh] - airspace_center_y) ** 2)
            distance_average[i] += distance
        distance_average[i] = distance_average[i] / airspace_cover_num_
    distance_average = list(distance_average)
    distance_shortest_index = distance_average.index(min(distance_average))
    airspace_best_bh = int(max_tgt_num_airspace_id[distance_shortest_index])

    """计算最优空域中心"""
    airspace_num_x = int((180 - scan_range_x) / 5 + 1)
    hang = np.fix(airspace_best_bh / airspace_num_x)  # 计算空域在第几行
    lie = airspace_best_bh - hang * airspace_num_x  # 计算空域在第几列
    airspace_center_x = -90.0 + scan_range_x / 2 + lie * 5.0
    airspace_center_y = 90.0 - scan_range_y / 2 - hang * 5.0
    # 更新目标覆盖情况
    target_cover_situation_new = deepcopy(target_cover_situation)
    for i in range(tgt_num):
        if (abs(x_m[i] - airspace_center_x) < scan_range_x / 2) and (abs(y_m[i] - airspace_center_y) < scan_range_y / 2):
            target_cover_situation_new[i] += 1
    # print(sum(target_cover_situation_new) - sum(target_cover_situation))

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

def plot_airspace(center_x, center_y, range_x, range_y, color, linestyle, label=''):
    plt.plot([center_x - range_x / 2, center_x + range_x / 2],
             [center_y - range_y / 2, center_y - range_y / 2], color=color, linestyle=linestyle, label=label)
    plt.plot([center_x + range_x / 2, center_x + range_x / 2],
             [center_y - range_y / 2, center_y + range_y / 2], color=color, linestyle=linestyle)
    plt.plot([center_x + range_x / 2, center_x - range_x / 2],
             [center_y + range_y / 2, center_y + range_y / 2], color=color, linestyle=linestyle)
    plt.plot([center_x - range_x / 2, center_x - range_x / 2],
             [center_y + range_y / 2, center_y - range_y / 2], color=color, linestyle=linestyle)
    plt.legend()

def plot_sub_airspace(Mi, scan_center, scan_range, color, label=''):
    scan_center_x, scan_center_y = scan_center[0], scan_center[1]
    scan_range_x, scan_range_y = scan_range[0], scan_range[1]
    """计算各子空域中心"""
    if Mi == 4:
        plt.plot([scan_center_x - scan_range_x / 2, scan_center_x + scan_range_x / 2], [scan_center_y, scan_center_y], color=color, linestyle='--', label=label)
        plt.plot([scan_center_x, scan_center_x], [scan_center_y - scan_range_y / 2, scan_center_y + scan_range_x / 2], color=color, linestyle='--')

def compute_beam_parameter(kongyu_num, Nsi, Mi, scan_range_x, scan_range_y, beamwidth, init_scan_range, tao_sij):
    beam_as_x, beam_as_y = [], []
    beam_sub_x, beam_sub_y = [], []
    beam_bh, beam_position = [], []
    sub_scan_range_x, sub_scan_range_y = [], []     # 子空域初始化搜索方位
    for i in range(kongyu_num):
        if Mi[i] == 1:
            sub_scan_range_x.append([init_scan_range[i][0][0]])   # 就一个空域
            sub_scan_range_y.append([init_scan_range[i][1][1]])
        else:
            sub_scan_range_x.append([init_scan_range[i][0][0], init_scan_range[i][0][0] + scan_range_x[i] / 2,
                                     init_scan_range[i][0][0], init_scan_range[i][0][0] + scan_range_x[i] / 2])
            sub_scan_range_y.append([init_scan_range[i][1][1], init_scan_range[i][1][1],
                    init_scan_range[i][1][1] - scan_range_y[i] / 2, init_scan_range[i][1][1] - scan_range_y[i] / 2])
    beam_tao_sij = []  # 各波位驻留时间
    for i in range(kongyu_num):
        beam_as_x.append(int(scan_range_x[i] / beamwidth[i]))
        beam_as_y.append(int(scan_range_y[i] / beamwidth[i]))
        beam_sub_x.append(int(scan_range_x[i] / beamwidth[i] / np.sqrt(Mi[i])))
        beam_sub_y.append(int(scan_range_y[i] / beamwidth[i] / np.sqrt(Mi[i])))
        _beam_bh, _beam_position, _beam_tao_sij = [], [], []
        for j in range(Mi[i]):
            __beam_bh = np.zeros((int(Nsi[i] / Mi[i]),))
            __beam_position = np.zeros((int(Nsi[i] / Mi[i]), 2))
            __beam_tao_sij = np.zeros((int(Nsi[i] / Mi[i]),))
            init_scan_x = sub_scan_range_x[i][j] + beamwidth[i] / 2.0
            init_scan_y = sub_scan_range_y[i][j] - beamwidth[i] / 2.0
            for k in range(int(Nsi[i] / Mi[i])):
                __beam_bh[k] = k
                hang = int(k / beam_sub_x[i])
                lie = k - hang * beam_sub_x[i]
                __beam_position[k][0] = init_scan_x + lie * beamwidth[i]
                __beam_position[k][1] = init_scan_y - hang * beamwidth[i]
                __beam_tao_sij[k] = tao_sij[i][j]
            _beam_bh.append(__beam_bh)
            _beam_position.append(__beam_position)
            _beam_tao_sij.append(__beam_tao_sij)
        beam_bh.append(_beam_bh)
        beam_position.append(_beam_position)
        beam_tao_sij.append(_beam_tao_sij)
    return beam_as_x, beam_as_y, beam_sub_x, beam_sub_y, beam_bh, beam_position, beam_tao_sij

def plot_circle(x, y, r, color, linestyle='-', linewidth='0.3'):
    # 画一个圆
    t = np.arange(0.0, 2 * np.pi, 0.1)
    X = r * np.cos(t) + x
    Y = r * np.sin(t) + y
    plt.plot(X, Y, color=color, linestyle=linestyle, linewidth=linewidth)

def f(mu, sigma, x, y):
    return 1.0 / (2.0 * np.pi * sigma**2) * np.exp(-((x - mu[0])**2 + (y - mu[1])**2) / (2.0 * sigma**2))

def compute_jifen(mu, sigma, x_min, x_max, y_min, y_max):
    buchang = 0.1
    jifen = 0
    x = np.arange(x_min + buchang, x_max + buchang, buchang)
    y = np.arange(y_min + buchang, y_max + buchang, buchang)
    for i in range(np.size(x)):
        for j in range(np.size(y)):
            jifen += f(mu, sigma, x[i], y[j]) * buchang**2    # 函数值乘以面积
    return jifen



glo_tgt_num = 30  # 目标总数目
glo_coo_num = 3   # 协同雷达数目
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
alpha_i = np.array([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0])    # 载机雷达权值
pd = [0.85, 0.9, 0.95]  # 检测概率
pfa = [2.0e-6, 1.5e-6, 1.0e-6]  # 虚警率
SNR_D = []
for i in range(N):
    SNR_D.append(np.log(pfa[i]) / np.log(pd[i]) - 1.0)
_omega = compute_omega(Pav, dB2W(Gt), dB2W(Gr), lamda, RCS_sigma, k, T0, dB2W(Fn), dB2W(L))
omega = [0.01 * _omega / 100.0, 0.05 * _omega / 100.0, 0.1 * _omega / 100.0]    # 雷达系统常数

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
print("Nsi", Nsi)  # 各子空域波位数：25，16，25

# 目标参数设置，这些威胁度全部是固定的，只有目标坐标是随机的
target_num = 30
sigma = 3  # 目标分布标准差
mb_type = np.arange(0.8, 1.0, 0.2 / target_num)  # 目标类型威胁度
v = np.arange(900.0, 1200.0, 300.0 / target_num)  # 目标速度，单位m/s
R = np.arange(200.0, 400.0, 200.0 / target_num)  # 目标距离，单位km
threat_w = compute_threat(target_num, mb_type, v, R)  # 每个目标的威胁度
sum_w_v = sum(threat_w * v)  # 计算平均速度

colors = ['purple', 'green', 'blue']

# 随机初始化目标坐标
# x_m = np.random.uniform(-60.0, 60.0, size=(target_num,))
# y_m = np.random.uniform(-60.0, 60.0, size=(target_num,))
# print(x_m)
# print(y_m)
x_m = np.array([ 59.042313,   -42.88954598,  38.67208861, -49.70329339, -45.26035556,
 -52.08528442,  49.55247246,  16.7500877,  -19.73814657, -45.76646825,
   2.37954842, -49.91596497,  22.4086236,  -52.84027131, -29.24156786,
  -7.6775505,   28.71594207,  -0.20446516, -34.35614678,  44.97092937,
 -39.36707867, -11.74322469,  37.29615862,  -6.37732823,  59.14980702,
 -38.74154256, -12.32077081, -27.15469921, -41.29885164, -11.85867341])
y_m = np.array([ -0.82467199,  43.85064069,  -0.31048872, -59.68637298,  36.79649007,
  22.07336598,   7.87693914,  52.50553277,   7.00903203,  -4.6598683,
  32.16021096, -35.55126643,  16.94456368,  13.55969979, -31.62187136,
   7.20915001,  38.82454103, -11.06718068,  57.53184837,  18.77611681,
   8.07406794, -44.20869572, -18.87051973,  29.7147276,  -40.44449416,
 -13.06059527, -34.00989031, -52.12344987, -39.18521733, -47.58097732])
# [ 59.042313   -42.88954598  38.67208861 -49.70329339 -45.26035556
#  -52.08528442  49.55247246  16.7500877  -19.73814657 -45.76646825
#    2.37954842 -49.91596497  22.4086236  -52.84027131 -29.24156786
#   -7.6775505   28.71594207  -0.20446516 -34.35614678  44.97092937
#  -39.36707867 -11.74322469  37.29615862  -6.37732823  59.14980702
#  -38.74154256 -12.32077081 -27.15469921 -41.29885164 -11.85867341]
# [ -0.82467199  43.85064069  -0.31048872 -59.68637298  36.79649007
#   22.07336598   7.87693914  52.50553277   7.00903203  -4.6598683
#   32.16021096 -35.55126643  16.94456368  13.55969979 -31.62187136
#    7.20915001  38.82454103 -11.06718068  57.53184837  18.77611681
#    8.07406794 -44.20869572 -18.87051973  29.7147276  -40.44449416
#  -13.06059527 -34.00989031 -52.12344987 -39.18521733 -47.58097732]
# 随机初始化目标发现状况
# target_cover_situation = np.zeros((target_num,))  # 空域覆盖状态，0，1对应位置目标是否被覆盖
# target_cover_situation_flag = np.random.uniform(0.0, 1.0, size=(target_num,))
# for i in range(target_num):
#     if target_cover_situation_flag[i] <= 0.3:
#         target_cover_situation[i] = 1
# target_cover_situation_init = deepcopy(target_cover_situation)
# print(target_cover_situation_init)
# target_cover_situation = np.array([1., 0., 1., 0., 1., 0., 1., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 0., 1., 1., 1.])
target_cover_situation = np.zeros((target_num,))
target_cover_situation_init = deepcopy(target_cover_situation)

# 进行一次空域划分，顺便计算betaij
betaij, all_betaij = [], []  # 首先计算各子空域betaij
tgt_numij, all_tgt_numij = [], []
cover_num, all_cover_num = [], []
cover_bh, all_cover_bh = [], []
airspace_center_x_record, airspace_center_y_record = [], []
all_x, all_y = [], []
all_kongyu_num = []
r_id = 0
label_flag = 0
plt.figure(2)
while min(target_cover_situation) <= 0:
    betaij, tgt_numij, cover_num, cover_bh, airspace_center_x_record, airspace_center_y_record = [], [], [], [], [], []
    for r_id in range(glo_coo_num):
        airspace_center_x, airspace_center_y, target_cover_situation = \
            greedy_algorithm(x_m, y_m, scan_range_x[r_id], scan_range_y[r_id], target_cover_situation)
        airspace_center_x_record.append(airspace_center_x)
        airspace_center_y_record.append(airspace_center_y)
        _cover_num = 0
        _cover_bh =[]
        for j in range(target_num):
            if abs(airspace_center_x - x_m[j]) < scan_range_x[r_id] / 2.0 and abs(airspace_center_y - y_m[j]) < scan_range_y[r_id] / 2.0:
                _cover_num += 1
                _cover_bh.append(j + 1)
        _betaij, _tgy_numij = compute_betaij(x_m, y_m, threat_w, Mi[r_id], [airspace_center_x, airspace_center_y], [scan_range_x[r_id], scan_range_y[r_id]])
        if label_flag == 0:
            plot_airspace(airspace_center_x, airspace_center_y, scan_range_x[r_id], scan_range_y[r_id], colors[r_id], '-', 'Agent {}'.format(r_id + 1))
            plot_sub_airspace(Mi[r_id], [airspace_center_x, airspace_center_y], [scan_range_x[r_id], scan_range_y[r_id]], colors[r_id], 'Agent {} sub-airspace boundary'.format(r_id + 1))
            if r_id == glo_coo_num - 1:
                label_flag = 1
        else:
            plot_airspace(airspace_center_x, airspace_center_y, scan_range_x[r_id], scan_range_y[r_id], colors[r_id], '-')
            plot_sub_airspace(Mi[r_id], [airspace_center_x, airspace_center_y], [scan_range_x[r_id], scan_range_y[r_id]], colors[r_id])
        cover_num.append(_cover_num)
        cover_bh.append(_cover_bh)
        betaij.append(_betaij)
        tgt_numij.append(_tgy_numij)
        if min(target_cover_situation) > 0:
            break
    all_kongyu_num.append(r_id + 1)
    all_betaij.append(betaij)
    all_tgt_numij.append(tgt_numij)
    all_cover_num.append(cover_num)
    all_cover_bh.append(cover_bh)
    all_x.append(airspace_center_x_record)
    all_y.append(airspace_center_y_record)
# print("all_cover_num", all_cover_num)
# print("all_betaij", all_betaij)
# print("all_tgt_numij", all_tgt_numij)
print("all_kongyu_num", all_kongyu_num)
plt.legend()
plt.axis("equal")
# plt.show()
# 二维空域可视化
flag1, flag2 = 0, 0
plt.figure(2)
for i in range(target_num):
    if target_cover_situation_init[i] == 0:
        if flag1 == 0:
            plt.plot(x_m[i], y_m[i], 'k^', label='target')
            flag1 = 1
        else:
            plt.plot(x_m[i], y_m[i], 'k^')
    else:
        if flag2 == 0:
            plt.plot(x_m[i], y_m[i], 'k^', label='discovered target')
            flag2 = 1
        else:
            plt.plot(x_m[i], y_m[i], 'k^')
    plot_circle(x_m[i], y_m[i], sigma, 'red', linestyle='-', linewidth='1.0')
plt.legend()
all_init_scan_range = []
for num in range(np.size(all_kongyu_num)):
    init_scan_range = []
    for i in range(all_kongyu_num[num]):
        scan_x_min = all_x[num][i] - scan_range_x[i] / 2.0
        scan_x_max = all_x[num][i] + scan_range_x[i] / 2.0
        scan_y_min = all_y[num][i] - scan_range_y[i] / 2.0
        scan_y_max = all_y[num][i] + scan_range_y[i] / 2.0
        init_scan_range.append([[scan_x_min, scan_x_max], [scan_y_min, scan_y_max]])
    all_init_scan_range.append(init_scan_range)
plt.axis('equal')
# plt.show()
all_R_DE_ba, all_R_DE_ba_duibi = [], []
all_R_DEi, all_R_DEi_duibi = [], []
all_P, all_P_duibi = [], []
all_P_record, all_P_duibi_record = [], []
for num in range(np.size(all_kongyu_num)):
    kongyu_num = all_kongyu_num[num]
    betaij = all_betaij[num]
    tgt_numij = all_tgt_numij[num]
    cover_num = all_cover_num[num]
    cover_bh = all_cover_bh[num]
    airspace_center_x_record = all_x[num]
    airspace_center_y_record = all_y[num]

    tao_sij = []
    tao_sij_avg = np.zeros((kongyu_num,))
    R_DEi = np.zeros((kongyu_num,))
    R_DEi_duibi = np.zeros((kongyu_num,))
    R_DEi_ba_record = []
    R_DEi_ba_duibi_record = []
    wvi = []
    tfi = np.zeros((kongyu_num,))
    for i in range(kongyu_num):
        tao_sj = np.zeros((Mi[i],))
        wv = 0.0
        w_new = []
        for j in range(cover_num[i]):
            tgt_bh = int(cover_bh[i][j] - 1)
            w_new.append(threat_w[tgt_bh])
        for j in range(cover_num[i]):
            tgt_bh = int(cover_bh[i][j] - 1)
            wv += w_new[j] / sum(w_new) * v[tgt_bh]
        sub_beam_num = int(Nsi[i] / Mi[i])
        for j in range(Mi[i]):
            tao_sj[j] = betaij[i][j] ** (4.0 / 3.0) * (omega[i] / SNR_D[i]) ** (1.0 / 3.0) / (2 * sub_beam_num * beilv * wv) ** (4.0 / 3.0)
        _tfi = sub_beam_num * beilv * sum(tao_sj)
        _R_DEi = sum(betaij[i] * ((omega[i] * tao_sj * 1.0 / SNR_D[i]) ** (1.0 / 4.0) - _tfi / 2 * wv)) / 1000.0
        tao_sij_avg[i] = sum(tao_sj) / Mi[i]
        _R_DEi_duibi = sum(betaij[i] * ((omega[i] * tao_sij_avg[i] * 1.0 / SNR_D[i]) ** (1.0 / 4.0) - _tfi / 2 * wv)) / 1000.0
        tao_sij.append(tao_sj)
        R_DEi[i] = _R_DEi
        R_DEi_duibi[i] = _R_DEi_duibi
        wvi.append(wv)
        tfi[i] = _tfi
    all_R_DE_ba.append(sum(alpha_i[:kongyu_num] / sum(alpha_i[:kongyu_num]) * R_DEi))
    all_R_DE_ba_duibi.append(sum(alpha_i[:kongyu_num] / sum(alpha_i[:kongyu_num]) * R_DEi_duibi))
    all_R_DEi.append(R_DEi)
    all_R_DEi_duibi.append(R_DEi_duibi)

    init_scan_range = all_init_scan_range[num]
    # 计算各波位中心极坐标，及各波位波束驻留时间
    beam_as_x, beam_as_y, beam_sub_x, beam_sub_y, beam_bh, beam_position, beam_tao_sij = \
        compute_beam_parameter(kongyu_num, Nsi, Mi, scan_range_x, scan_range_y, beamwidth, init_scan_range, tao_sij)
    # 二维波位可视化
    plt.figure(2)
    for i in range(kongyu_num):
        for j in range(Mi[i]):
            for k in range(int(Nsi[i] / Mi[i])):
                # if j == 0:
                plot_circle(beam_position[i][j][k][0], beam_position[i][j][k][1], beamwidth[i] / 2.0, colors[i])
    # plt.show()
    pijk, nijk, nijk_duibi = [], [], []
    for i in range(kongyu_num):
        pjk, njk, njk_duibi = [], [], []
        sub_beam_num = int(Nsi[i] / Mi[i])
        for j in range(Mi[i]):
            pk = np.zeros((target_num, sub_beam_num))
            pk_sum = np.zeros((sub_beam_num,))
            for h in range(target_num):
                for k in range(sub_beam_num):
                    beam_center_x = beam_position[i][j][k][0]
                    beam_center_y = beam_position[i][j][k][1]
                    pk[h][k] = compute_jifen([x_m[h], y_m[h]], sigma, beam_center_x - beamwidth[i] / 2,
                                             beam_center_x + beamwidth[i] / 2, beam_center_y - beamwidth[i] / 2,
                                             beam_center_y + beamwidth[i] / 2)
            for h in range(target_num):
                if int(h + 1) in cover_bh[i]:
                    pk_sum += pk[h]
            pjk.append(pk_sum)
            nk = np.zeros((sub_beam_num,))
            log_sum = 0.0
            for k in range(sub_beam_num):
                log_sum += np.log(pk_sum[0] / pk_sum[k]) / np.log(1 - pd[i])
            for k in range(sub_beam_num):
                nk[k] = np.log(pk_sum[0] / pk_sum[k]) / np.log(1 - pd[i]) + 1.0 / sub_beam_num * (betaij[i][j] * tfi[i] / tao_sij[i][j] - log_sum)
            # print(sum(nk), betaij[i][j] * tfi[i] / tao_sij[i][j])   # 这两个参数应该是相等的，毕竟是约束条件
            njk.append(nk)
        sum_beam_num = 0.0
        for j in range(Mi[i]):
            sum_beam_num += sum(njk[j])
        pijk.append(pjk)
        nijk.append(njk)
        nijk_duibi.append(sum_beam_num / Nsi[i])

    P = np.zeros((kongyu_num,))
    P_duibi = np.zeros((kongyu_num,))
    for i in range(kongyu_num):
        for j in range(Mi[i]):
            sub_beam_num = int(Nsi[i] / Mi[i])
            for k in range(sub_beam_num):
                P[i] += pijk[i][j][k] / sum(cover_num) * (1 - (1 - pd[i]) ** (nijk[i][j][k] * 1.0))
                P_duibi[i] += pijk[i][j][k] / sum(cover_num) * (1 - (1 - pd[i]) ** (beilv * 1.0))
    all_P_record.append(P)
    all_P_duibi_record.append(P_duibi)
    all_P.append(sum(P))
    all_P_duibi.append(sum(P_duibi))

print("all_R_DE_ba", all_R_DE_ba)
plt.figure(1)
x_label = np.arange(1, np.size(all_R_DE_ba) + 1, 1)
plt.plot(x_label, all_R_DE_ba, 'r^', linestyle='-', label='coo-opt')
plt.plot(x_label, all_R_DE_ba_duibi, 'ro', linestyle='--', label='coo-ori')
plt.xlabel('No. of cooperative search')
plt.ylabel('cooperative expected discovery distance of cluster targets')
plt.legend()
# plt.show()

print("P", all_P, "P_duibi", all_P_duibi)
x_label = np.arange(1, np.size(all_P) + 1, 1)
plt.figure(3)
plt.plot(x_label, all_P, 'r^', linestyle='-', label='coo-opt')
plt.plot(x_label, all_P_duibi, 'ro', linestyle='--', label='coo-ori')
plt.xlabel('No. of cooperative search')
plt.ylabel('cooperative accumulated discovery probability of cluster targets')
plt.legend()
# plt.show()

all_R_DEi = np.array(all_R_DEi)
all_R_DEi_duibi = np.array(all_R_DEi_duibi)
plt.figure(1)
x_label = np.arange(1, np.size(all_R_DE_ba) + 1, 1)
for i in range(glo_coo_num):
    plt.plot(x_label, all_R_DEi[:, i], color=colors[i], marker='^', linestyle='-', label='r_{}_opt'.format(i + 1))
    plt.plot(x_label, all_R_DEi_duibi[:, i], color=colors[i], marker='o', linestyle='--', label='r_{}_ori'.format(i + 1))
# plt.xlabel('No. of cooperative search')
# plt.ylabel('radar expected discovery distance of cluster targets')
plt.legend()
# plt.show()

all_P_record = np.array(all_P_record)
all_P_duibi_record = np.array(all_P_duibi_record)
plt.figure(3)
x_label = np.arange(1, np.size(all_R_DE_ba) + 1, 1)
for i in range(glo_coo_num):
    plt.plot(x_label, all_P_record[:, i], color=colors[i], marker='^', linestyle='-', label='r_{}_opt'.format(i + 1))
    plt.plot(x_label, all_P_duibi_record[:, i], color=colors[i], marker='o', linestyle='--', label='r_{}_ori'.format(i + 1))
# plt.xlabel('No. of cooperative search')
# plt.ylabel('radar accumulated discovery probability of cluster targets')
plt.legend()
plt.show()


















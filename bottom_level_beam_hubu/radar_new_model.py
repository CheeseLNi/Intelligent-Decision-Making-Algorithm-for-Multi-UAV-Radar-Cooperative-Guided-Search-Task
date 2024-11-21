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
    if Mi == 4:
        plt.plot([scan_center_x - scan_range_x / 2, scan_center_x + scan_range_x / 2], [scan_center_y, scan_center_y], color=color, linestyle='--')
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
alpha_i = [0.2, 0.3, 0.5]    # 载机雷达权值
pd = [0.85, 0.9, 0.95]  # 检测概率
pfa = [2.0e-6, 1.5e-6, 1.0e-6]  # 虚警率
SNR_D = []
for i in range(N):
    SNR_D.append(np.log(pfa[i]) / np.log(pd[i]) - 1.0)
_omega = compute_omega(Pav, dB2W(Gt), dB2W(Gr), lamda, RCS_sigma, k, T0, dB2W(Fn), dB2W(L))
omega = [0.01 * _omega / 10.0, 0.05 * _omega / 10.0, 0.1 * _omega / 10.0]    # 雷达系统常数

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
x_m = np.random.uniform(-60.0, 60.0, size=(target_num,))
y_m = np.random.uniform(-60.0, 60.0, size=(target_num,))
# 随机初始化目标发现状况
target_cover_situation = np.zeros((target_num,))  # 空域覆盖状态，0，1对应位置目标是否被覆盖
target_cover_situation_flag = np.random.uniform(0.0, 1.0, size=(target_num,))
for i in range(target_num):
    if target_cover_situation_flag[i] <= 0.5:
        target_cover_situation[i] = 1
target_cover_situation_init = deepcopy(target_cover_situation)

# 进行一次空域划分，顺便计算betaij
betaij = []  # 首先计算各子空域betaij
tgt_numij = []
cover_num = []
cover_bh = []
airspace_center_x_record, airspace_center_y_record = [], []
num_0 = 0
plt.figure(2)
while num_0 < glo_coo_num and min(target_cover_situation) <= 0:
    airspace_center_x, airspace_center_y, target_cover_situation = \
        greedy_algorithm(x_m, y_m, scan_range_x[num_0], scan_range_y[num_0], target_cover_situation)
    airspace_center_x_record.append(airspace_center_x)
    airspace_center_y_record.append(airspace_center_y)
    _cover_num = 0
    _cover_bh =[]
    for j in range(target_num):
        if abs(airspace_center_x - x_m[j]) < scan_range_x[num_0] / 2.0 and abs(airspace_center_y - y_m[j]) < scan_range_y[num_0] / 2.0:
            _cover_num += 1
            _cover_bh.append(j + 1)
    _betaij, _tgy_numij = compute_betaij(x_m, y_m, threat_w, Mi[num_0], [airspace_center_x, airspace_center_y], [scan_range_x[num_0], scan_range_y[num_0]])
    plot_airspace(airspace_center_x, airspace_center_y, scan_range_x[num_0], scan_range_y[num_0], colors[num_0], '-')
    plot_sub_airspace(Mi[num_0], [airspace_center_x, airspace_center_y], [scan_range_x[num_0], scan_range_y[num_0]], colors[num_0])
    cover_num.append(_cover_num)
    cover_bh.append(_cover_bh)
    betaij.append(_betaij)
    tgt_numij.append(_tgy_numij)
    num_0 += 1
print("cover_num", cover_num)
print("betaij", betaij)
print("tgt_numij", tgt_numij)
kongyu_num = np.size(airspace_center_x_record)
init_scan_range = []
# 二维空域可视化
plt.figure(2)
for i in range(target_num):
    if target_cover_situation_init[i] == 0:
        plt.plot(x_m[i], y_m[i], 'r^', label='target')
    else:
        plt.plot(x_m[i], y_m[i], 'k^', label='target')
for i in range(kongyu_num):
    scan_x_min = airspace_center_x_record[i] - scan_range_x[i] / 2.0
    scan_x_max = airspace_center_x_record[i] + scan_range_x[i] / 2.0
    scan_y_min = airspace_center_y_record[i] - scan_range_y[i] / 2.0
    scan_y_max = airspace_center_y_record[i] + scan_range_y[i] / 2.0
    init_scan_range.append([[scan_x_min, scan_x_max], [scan_y_min, scan_y_max]])
plt.axis('equal')
# plt.show()
# 计算R_DE
tao_sij = []
tao_sij_avg = np.zeros((kongyu_num,))
R_DEi = np.zeros((kongyu_num,))
R_DEi_duibi = np.zeros((kongyu_num,))
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
        tao_sj[j] = betaij[i][j]**(4.0/3.0) * (omega[i] / SNR_D[i])**(1.0/3.0) / (2 * sub_beam_num * beilv * wv)**(4.0/3.0)
    _tfi = sub_beam_num * beilv * sum(tao_sj)
    _R_DEi = sum(betaij[i] * ((omega[i] * tao_sj / SNR_D[i])**(1.0/4.0) - _tfi / 2 * wv)) / 1000.0
    tao_sij_avg[i] = sum(tao_sj) / Mi[i]
    _R_DEi_duibi = sum(betaij[i] * ((omega[i] * tao_sij_avg[i] / SNR_D[i])**(1.0/4.0) - _tfi / 2 * wv)) / 1000.0
    tao_sij.append(tao_sj)
    R_DEi[i] = _R_DEi
    R_DEi_duibi[i] = _R_DEi_duibi
    wvi.append(wv)
    tfi[i] = _tfi
print("tao_sij", tao_sij)
print("R_DEi", R_DEi, "R_DEi_duibi", R_DEi_duibi)
print("betaij * tfi / taosij", sum(betaij[0] * tfi[0] / tao_sij[0]), sum(betaij[1] * tfi[1] / tao_sij[1]), sum(betaij[2] * tfi[2] / tao_sij[2]))
# print(wvi)
print("tfi", tfi)
R_DE_ba = sum(alpha_i * R_DEi)
R_DE_ba_duibi = sum(alpha_i * R_DEi_duibi)
print("R_DE_ba", R_DE_ba, "R_DE_ba_duibi", R_DE_ba_duibi)
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
                    beam_center_x + beamwidth[i] / 2, beam_center_y - beamwidth[i] / 2, beam_center_y + beamwidth[i] / 2)
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

P = 0.0
P_duibi = 0.0
for i in range(kongyu_num):
    for j in range(Mi[i]):
        sub_beam_num = int(Nsi[i] / Mi[i])
        for k in range(sub_beam_num):
            P += pijk[i][j][k] / sum(cover_num) * (1 - (1 - pd[i]) ** nijk[i][j][k])
            P_duibi += pijk[i][j][k] / sum(cover_num) * (1 - (1 - pd[i]) ** beilv)
print("P", P, "P_duibi", P_duibi)
plt.show()



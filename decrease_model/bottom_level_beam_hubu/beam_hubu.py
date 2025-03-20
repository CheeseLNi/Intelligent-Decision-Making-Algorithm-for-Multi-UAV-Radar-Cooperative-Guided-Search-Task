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

def compute_beam_parameter(kongyu_num, Nsi, Mi, scan_range_x, scan_range_y, beamwidth, init_scan_range):
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
                # __beam_tao_sij[k] = tao_sij[i][j]
            _beam_bh.append(__beam_bh)
            _beam_position.append(__beam_position)
            # _beam_tao_sij.append(__beam_tao_sij)
        beam_bh.append(_beam_bh)
        beam_position.append(_beam_position)
        beam_tao_sij.append(_beam_tao_sij)
    return beam_as_x, beam_as_y, beam_sub_x, beam_sub_y, beam_bh, beam_position

def plot_circle(x, y, r, color, linestyle='-', linewidth='0.3', label=''):
    # 画一个圆
    t = np.arange(0.0, 2 * np.pi, 0.1)
    X = r * np.cos(t) + x
    Y = r * np.sin(t) + y
    plt.plot(X, Y, color=color, linestyle=linestyle, linewidth=linewidth, label=label)

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

def track_marker(x, y, z, color):
    r = 5.0
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

    # plt.pause(0.01)

def display_single_beam_no_remove(center_x, center_y, beamwidth, Rd, colors):
    # 求四个边角点
    azimuth = (center_x + beamwidth / 2.0) * np.pi / 180.0
    pitch = (center_y + beamwidth / 2.0) * np.pi / 180.0
    xt_1 = Rd * np.cos(pitch) * np.cos(azimuth)
    yt_1 = Rd * np.cos(pitch) * np.sin(azimuth)
    zt_1 = Rd * np.sin(pitch)

    azimuth = (center_x - beamwidth / 2.0) * np.pi / 180.0
    pitch = (center_y + beamwidth / 2.0) * np.pi / 180.0
    xt_2 = Rd * np.cos(pitch) * np.cos(azimuth)
    yt_2 = Rd * np.cos(pitch) * np.sin(azimuth)
    zt_2 = Rd * np.sin(pitch)

    azimuth = (center_x - beamwidth / 2.0) * np.pi / 180.0
    pitch = (center_y - beamwidth / 2.0) * np.pi / 180.0
    xt_3 = Rd * np.cos(pitch) * np.cos(azimuth)
    yt_3 = Rd * np.cos(pitch) * np.sin(azimuth)
    zt_3 = Rd * np.sin(pitch)

    azimuth = (center_x + beamwidth / 2.0) * np.pi / 180.0
    pitch = (center_y - beamwidth / 2.0) * np.pi / 180.0
    xt_4 = Rd * np.cos(pitch) * np.cos(azimuth)
    yt_4 = Rd * np.cos(pitch) * np.sin(azimuth)
    zt_4 = Rd * np.sin(pitch)

    line1 = plt.plot([xt_1, xt_2], [yt_1, yt_2], [zt_1, zt_2], color=colors, linestyle='-')
    line2 = plt.plot([xt_2, xt_3], [yt_2, yt_3], [zt_2, zt_3], color=colors, linestyle='-')
    line3 = plt.plot([xt_3, xt_4], [yt_3, yt_4], [zt_3, zt_4], color=colors, linestyle='-')
    line4 = plt.plot([xt_4, xt_1], [yt_4, yt_1], [zt_4, zt_1], color=colors, linestyle='-')

    line5 = plt.plot([xt_1, 0.0], [yt_1, 0.0], [zt_1, 0.0], color=colors, linestyle='--')
    line6 = plt.plot([xt_2, 0.0], [yt_2, 0.0], [zt_2, 0.0], color=colors, linestyle='--')
    line7 = plt.plot([xt_3, 0.0], [yt_3, 0.0], [zt_3, 0.0], color=colors, linestyle='--')
    line8 = plt.plot([xt_4, 0.0], [yt_4, 0.0], [zt_4, 0.0], color=colors, linestyle='--')

    # lines.append([line5[0], line6[0], line7[0], line8[0]])
    # lines = [line1[0], line2[0], line3[0], line4[0], line5[0], line6[0], line7[0], line8[0]]
    # plt.pause(0.01)
    # for line in lines:
    #     line.remove()

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

Rm = np.random.uniform(150.0, 170.0, (target_num,))
x_m_3d = Rm * np.cos(y_m * np.pi / 180.0) * np.cos(x_m * np.pi / 180.0)
y_m_3d = Rm * np.cos(y_m * np.pi / 180.0) * np.sin(x_m * np.pi / 180.0)
z_m_3d = Rm * np.sin(y_m * np.pi / 180.0)
plt.ion()
plt.figure(4)
plt.subplot(projection='3d')
# plt.plot(x_m_3d, y_m_3d, z_m_3d, 'k^')
# plt.axis('equal')

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
    if i == 0:
        plot_circle(x_m[i], y_m[i], 2 * sigma, 'red', linestyle='-', linewidth='1.0', label='target distibution error circle')
    else:
        plot_circle(x_m[i], y_m[i], 2 * sigma, 'red', linestyle='-', linewidth='1.0')
plt.legend()
all_init_scan_range = []
plt.figure(4)
for num in range(np.size(all_kongyu_num)):
    init_scan_range = []
    for i in range(all_kongyu_num[num]):
        scan_x_min = all_x[num][i] - scan_range_x[i] / 2.0
        scan_x_max = all_x[num][i] + scan_range_x[i] / 2.0
        scan_y_min = all_y[num][i] - scan_range_y[i] / 2.0
        scan_y_max = all_y[num][i] + scan_range_y[i] / 2.0
        init_scan_range.append([[scan_x_min, scan_x_max], [scan_y_min, scan_y_max]])
        # if num == 1:
        #     plot_airspace(all_x[num][i], all_y[num][i], scan_range_x[i], scan_range_y[i], 'blue', '-', 'airspace boundary')
        # else:
        #     plot_airspace(all_x[num][i], all_y[num][i], scan_range_x[i], scan_range_y[i], 'blue', '-')
    all_init_scan_range.append(init_scan_range)
plt.legend()
plt.axis('equal')
# plt.show()

kongyu_num = all_kongyu_num[0]
init_scan_range = all_init_scan_range[0]
Rd = 180
###################################################三维空域可视化######################################################
angle_point = []  # 空域边角三维坐标
plt.figure(4)
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
        # real_time[i] += beam_tao_sij[int(search_airpace_id[i])][int(search_beam_id[i])]
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
                    plt.figure(4)
                    track_marker(x_m_3d[j], y_m_3d[j], z_m_3d[j], 'red')
                    # plt.figure(2)
                    # track_marker_2d(x_m[j], y_m[j], colors[i])
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
                        plt.figure(4)
                        track_marker(x_m_3d[j], y_m_3d[j], z_m_3d[j], 'red')
                        # plt.figure(2)
                        # track_marker_2d(x_m[j], y_m[j], colors[i])
        if search_beam_id[i] >= beam_num_x[int(search_airpace_id[i])] * beam_num_y[int(search_airpace_id[i])] - 1:
            done[int(search_airpace_id[i])] = 1
            for j in range(kongyu_num):
                if done[j] == 0:
                    print(real_time)
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
        plt.figure(4)
        display_xietong_beam(xietong_search_x, xietong_search_y, beamwidth, Rd, colors, kongyu_num)
        plt.axis('equal')
        # plt.figure(2)
        # display_xietong_beam_2d(xietong_search_x, xietong_search_y, beamwidth, colors, kongyu_num)

print(real_time)







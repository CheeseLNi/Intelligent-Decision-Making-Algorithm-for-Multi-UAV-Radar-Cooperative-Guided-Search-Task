import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append("..")

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
    airspace_cover_target_ = compute_airspace_cover_target_(airspace_cover_target, airspace_cover_num, airspace_best_bh)
    airspace_cover_num_ = int(airspace_cover_num[airspace_best_bh])
    for i in range(airspace_cover_num_):
        target_bh = int(airspace_cover_target[airspace_best_bh, i] - 1)
        target_cover_situation[target_bh] += 1

    return airspace_center_x, airspace_center_y, target_cover_situation

target_num = 30      # 协同搜索目标为30
xietong_num = 3      # 载机雷达智能体数目
scan_range_x = [10, 20, 30]  # 方位搜索范围
scan_range_y = [10, 20, 30]  # 俯仰搜索范围
n_epochs = int(1e5)  # 采1e5组数据
input_record, output_record = [], []
# 初始化存储文件
log_Agent1_input = open("log_Agent1_input.csv", "w+")
log_Agent1_output = open("log_Agent1_output.csv", "w+")
log_Agent2_input = open("log_Agent2_input.csv", "w+")
log_Agent2_output = open("log_Agent2_output.csv", "w+")
log_Agent3_input = open("log_Agent3_input.csv", "w+")
log_Agent3_output = open("log_Agent3_output.csv", "w+")

log_Multi_input = [log_Agent1_input, log_Agent2_input, log_Agent3_input]
log_Multi_output = [log_Agent1_output, log_Agent2_output, log_Agent3_output]

for epoch in range(n_epochs):
    if epoch % 100 == 0:
        print(epoch)
    # 随机初始化目标坐标信息
    x_m = np.random.uniform(-60, 60, size=(target_num,))
    y_m = np.random.uniform(-60, 60, size=(target_num,))

    """计算未覆盖目标集合"""
    target_cover_situation = np.zeros((target_num,))  # 空域覆盖状态，0，1对应位置目标是否被覆盖
    global_target_uncover = []
    for i in range(target_num):
        if target_cover_situation[i] == 0:  # 说明目标未被覆盖
            target_bh = int(i + 1)
            global_target_uncover.append(target_bh)
    obs = np.zeros((1, 2 * target_num))
    for i in range(np.size(global_target_uncover)):
        target_bh = int(global_target_uncover[i] - 1)
        obs[0, 2 * i] = x_m[target_bh] / 60.0
        obs[0, 2 * i + 1] = y_m[target_bh] / 60.0

    """主循环"""
    while np.size(global_target_uncover) > 0:
        # 顺序执行每个智能体的搜索
        for i in range(xietong_num):
            # 保存观测，即输入
            for j in range(2 * target_num):
                log_Multi_input[i].write('{:.4f},'.format(obs[0, j]))
                log_Multi_input[i].flush()
            log_Multi_input[i].write('\n')

            # 计算对应搜索中心，并更新目标发现状态
            airspace_center_x, airspace_center_y, target_cover_situation = \
                compute_airspace_center_xy(x_m, y_m, scan_range_x[i], scan_range_y[i], target_cover_situation)

            # 保存搜索空域中心，即输出
            log_Multi_output[i].write('{:.4f}, {:.4f}\n'.format(airspace_center_x / 60.0, airspace_center_y / 60.0))
            log_Multi_output[i].flush()

            # 更新状态
            global_target_uncover = []
            for j in range(target_num):
                if target_cover_situation[j] == 0:  # 说明目标未被覆盖
                    target_bh = int(j + 1)
                    global_target_uncover.append(target_bh)
            obs = np.zeros((1, 2 * target_num))
            for j in range(np.size(global_target_uncover)):
                target_bh = int(global_target_uncover[j] - 1)
                obs[0, 2 * j] = x_m[target_bh] / 60.0
                obs[0, 2 * j + 1] = y_m[target_bh] / 60.0

            # print(global_target_uncover, np.size(global_target_uncover))
            # plt.figure()
            # plt.axis("equal")
            # plt.plot(x_m, y_m, '.', color='red')
            # plt.plot([airspace_center_x - scan_range_x[i] / 2, airspace_center_x + scan_range_x[i] / 2],
            #          [airspace_center_y - scan_range_y[i] / 2, airspace_center_y - scan_range_y[i] / 2], color='blue')
            # plt.plot([airspace_center_x + scan_range_x[i] / 2, airspace_center_x + scan_range_x[i] / 2],
            #          [airspace_center_y - scan_range_y[i] / 2, airspace_center_y + scan_range_y[i] / 2], color='blue')
            # plt.plot([airspace_center_x + scan_range_x[i] / 2, airspace_center_x - scan_range_x[i] / 2],
            #          [airspace_center_y + scan_range_y[i] / 2, airspace_center_y + scan_range_y[i] / 2], color='blue')
            # plt.plot([airspace_center_x - scan_range_x[i] / 2, airspace_center_x - scan_range_x[i] / 2],
            #          [airspace_center_y + scan_range_y[i] / 2, airspace_center_y - scan_range_y[i] / 2], color='blue')
            # plt.show()

            if np.size(global_target_uncover) == 0:
                break









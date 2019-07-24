import numpy as np
from matplotlib import pyplot as plt
from scipy import interpolate


# plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
# plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_one(action, episode) -> dict:
    with open('../results/no-optim-time-delays-%d-%d.txt' % (action, episode), 'r') as f:
        d = eval(f.readline())
        return d


def get_all_no_optim():
    a = []
    # for i in [2, 5, 8]:
    #     cur = {}
    #     for j in range(1):
    #         d = get_one(i, j)
    #         for k, v in d.items():
    #             if k in cur:
    #                 cur[k] += v
    #             else:
    #                 cur[k] = v
    #     for k, v in cur.items():
    #         cur[k] /= 1
    #     a.append(cur)

    cur = {}
    for i in range(1):
        with open('../results/optim-time-delays-%d.txt' % i, 'r') as f:
            d = eval(f.readline())
        for k, v in d.items():
            if k in cur:
                cur[k] += v
            else:
                cur[k] = v

        for k, v in cur.items():
            cur[k] /= 1
    a.append(cur)

    draw(a)


def draw(a: list, inter: bool = True):
    # plt.title("作业时延（每三分钟统计一次）")
    # plt.xlabel("时间（分钟）")
    # plt.ylabel("两分钟内完成的作业的时延（毫秒）")
    plt.title('Time Delay Every 3 Minutes')
    plt.xlabel("Time (Minutes)")
    plt.ylabel("Time Delay (Milliseconds)")
    num_items = max([len(item.keys()) for item in a])
    # num_items = 21
    x = [i * 3 for i in range(num_items)]

    # legends = ['Fixed Configuration 2', 'Fixed Configuration 5', 'Fixed Configuration 8', 'Optimized']
    legends = ['Optimized']
    for i, one in enumerate(a):
        y = [0] * num_items
        for j, v in enumerate(one.values()):
            # if j < 21:
            y[j] = v
            # else:
            #     break
        if inter:
            func = interpolate.interp1d(x, y, kind='cubic')
            x_new = np.arange(0, 57.1, 0.1)
            y_smooth = func(x_new)
            plt.plot(x_new, y_smooth, label=legends[i])
        else:
            plt.plot(x, y, label=legends[i])
    plt.legend()
    plt.show()


get_all_no_optim()

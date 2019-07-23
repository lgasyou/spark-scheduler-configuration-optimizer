from matplotlib import pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def get_one(action, episode) -> dict:
    with open('../results/no-optim-time-delays-%d-%d.txt' % (action, episode), 'r') as f:
        d = eval(f.readline())
        return d


def get_all_no_optim():
    a = []
    for i in range(4):
        cur = {}
        for j in range(3):
            d = get_one(i, j)
            for k, v in d.items():
                if k in cur:
                    cur[k] += v
                else:
                    cur[k] = v
        for k, v in cur.items():
            cur[k] /= 3
        a.append(cur)

    cur = {}
    for i in range(3):
        with open('../results/optim-time-delays-%d.txt' % i, 'r') as f:
            d = eval(f.readline())
        for k, v in d.items():
            if k in cur:
                cur[k] += v
            else:
                cur[k] = v

        for k, v in cur.items():
            cur[k] /= 3
    a.append(cur)

    draw(a)


def draw(a: list):
    plt.title("时延")
    plt.xlabel("时间（分钟）")
    plt.ylabel("两分钟内完成的作业的时延（毫秒）")
    num_items = max([len(item.keys()) for item in a])
    x = [i * 2 for i in range(num_items)]

    legends = ['固定配置项%d时延' % (i + 1) for i in range(len(a) - 1)]
    legends.append('优化后的时延')
    for i, one in enumerate(a):
        y = [0] * num_items
        for j, v in enumerate(one.values()):
            y[j] = v
        plt.plot(x, y, label=legends[i])
    plt.legend()
    plt.show()


get_all_no_optim()

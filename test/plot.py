from matplotlib import pyplot as plt


def get_one(action, episode) -> dict:
    with open('../results/no-optim-time-delays-%d-%d.txt' % (action, episode), 'r') as f:
        d = eval(f.readline())
        return d


def get_all_no_optim():
    a = []
    for i in range(3):
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

    with open('../results/optim-time-delays-0.txt', 'r') as f:
        d = eval(f.readline())
        a.append(d)

    draw(a)


def draw(a: list):
    plt.title("Time Delays")
    plt.xlabel("Time")
    plt.ylabel("Time delay of jobs which finished in 2 minutes")
    num_items = max([len(item.keys()) for item in a])
    x = [i * 2 for i in range(num_items)]

    legends = ['Fixed Action %d' % i for i in range(len(a) - 1)]
    legends.append('Optimized')
    for i, one in enumerate(a):
        y = [0] * num_items
        for j, v in enumerate(one.values()):
            y[j] = v
        plt.plot(x, y, label=legends[i])
    plt.legend()
    plt.show()


get_all_no_optim()

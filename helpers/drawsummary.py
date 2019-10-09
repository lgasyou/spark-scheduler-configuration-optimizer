import matplotlib.pyplot as plt


FILENAMES = ['../results/no-optim-delays-%d-0.txt' % i for i in range(3)]
FILENAMES.append('../results/optim-delays-0.txt')

legends = ['Fixed Configuration 0(25, 75)',
           'Fixed Configuration 1(50, 50)',
           'Fixed Configuration 2(75, 25)',
           'Optimized']


for idx, filename in enumerate(FILENAMES):
    with open(filename) as f:
        l = eval(f.readline())
        index, values = [], []
        for i in l:
            key = list(i.keys())[0]
            value = list(i.values())[0]
            if value:
                index.append(key)
                values.append(value)
        plt.plot(index, values, label=legends[idx])
    plt.legend()

plt.show()

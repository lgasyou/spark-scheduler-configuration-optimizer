import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate


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
        new_index = np.linspace(min(index), max(index), 5000)
        smooth_func = interpolate.interp1d(index, values, kind='cubic')
        smooth = smooth_func(new_index)
        plt.plot(new_index, smooth, label=legends[idx])
    plt.legend()

plt.show()

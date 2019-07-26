from matplotlib import pyplot as plt

from optimizer.util import fileutil


def log_loss(loss: float, filename: str = None):
    with open(filename, 'a') as f:
        f.write(str(loss) + '\n')


fileutil.log_into_file(666, '../results/losses.txt')

with open('../results/losses.txt', 'r') as f:
    losses = list(map(lambda item: float(item), f.readlines()))
    plt.plot([i for i in range(len(losses))], losses)
    plt.show()

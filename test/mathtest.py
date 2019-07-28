import math
from matplotlib import pyplot as plt

x = [i for i in range(400)]
factor = [0.5 * (1 + math.tanh((x - 200) / 200)) for x in range(400)]
y = [1.] * 400
for i in range(1, len(y)):
    y[i] = y[i - 1] * (1 + factor[i])

plt.plot(x, y)
plt.show()

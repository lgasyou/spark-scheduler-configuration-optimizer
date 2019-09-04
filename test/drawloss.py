from matplotlib import pyplot as plt


plt.rcParams['figure.dpi'] = 200

with open('../results/steps.txt', 'r') as f:
    times = []
    actions = []
    rewards = []
    for l in f.readlines():
        s = l.split(',')
        times.append(int(s[0]))
        actions.append(int(s[1]))
        rewards.append(float(s[2]))

with open('../results/losses.txt', 'r') as f:
    loss_times = []
    losses = []
    for l in f.readlines():
        s = l.split(',')
        loss_times.append(int(s[0]))
        losses.append(float(s[1]))

plt.subplot(2, 1, 1)
plt.title('Rewards')
plt.plot(times, rewards)

plt.subplot(2, 1, 2)
plt.title('Actions')
plt.plot(times, actions)

plt.savefig('../results/fig-%d-RA.png' % times[-1], format='png')
plt.show()

plt.title('Loss')
plt.plot(loss_times, losses)
plt.savefig('../results/fig-%d-L.png' % loss_times[-1], format='png')
plt.show()

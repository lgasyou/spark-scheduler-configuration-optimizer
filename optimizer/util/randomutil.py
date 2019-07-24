import random


def choice_p(items: list, p: list):
    h = [0] * len(p)
    h[0] = p[0]
    for i in range(1, len(p)):
        h[i] = h[i - 1] + p[i]

    p = random.randint(0, sum(p) - 1)
    for index, i in enumerate(h):
        if p < i:
            return items[index]

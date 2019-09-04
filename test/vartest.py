class H(object):

    def __init__(self):
        self.t = 666


def normalize(value):
    value /= 100


h = H()
print(h.t)
normalize(h.t)
print(h.t)

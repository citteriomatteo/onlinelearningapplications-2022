import numpy as np
from matplotlib import pyplot as plt


class CUSUM:
    def __init__(self, M, eps, h):
        self.M = M
        self.eps = eps
        self.h = h
        self.t = 1
        self.reference = 0
        self.g_plus = 0
        self.g_minus = 0

    def update(self, sample):
        self.t += 1
        if self.t <= self.M:
            self.reference += sample/self.M
            return False
        else:
            self.reference = (self.reference*(self.t-1) + sample)/self.t
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.g_minus = 0
        self.g_plus = 0
        self.t = 0
        self.reference = 0

p0 = 0.6
p1 = 0.2

eps = 0.03
M = 200
T = 1000
h = np.log(T)*2
CD = CUSUM(M, eps, h)
S = []
n_exp = 100
for i in range(n_exp):
    CD.reset()
    for t in range(T):
        p = p0 if t < 500 else p1
        sample = np.random.binomial(1, p)
        if CD.update(sample):
            print("GMINUS: ", CD.g_plus)
            print("GPLUS: ", CD.g_minus)
            print("Reference: ",CD.reference)
            print("Sample:",sample)
            print("Iteration:", i)
            S.append(t)
            break

plt.figure(0)
plt.hist(S, bins=20)
plt.figure(1)
plt.show()

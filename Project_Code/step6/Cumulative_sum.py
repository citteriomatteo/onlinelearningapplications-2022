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
        #empirical mean over M, definition of the reference point
        if self.t <= self.M:
            self.reference += sample/self.M
            return False
        else:
            self.reference = (self.reference*(self.t-1) + sample)/self.t
            #positive and negative deviation from reference point
            s_plus = (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps
            #positive and negative cumulative deviation from reference point
            self.g_plus = max(0, self.g_plus + s_plus)
            self.g_minus = max(0, self.g_minus + s_minus)
            #return 1 if the cumulative deviations exceed the threshold
            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.g_minus = 0
        self.g_plus = 0
        self.t = 0
        self.reference = 0
import numpy as np

class CUMSUM:
    def __init__(self,M,eps,h):
        self.M=M
        self.eps=eps
        self.h = h
        self.t=0
        self.reference=0
        self.g_plus=0
        self.g_minus=0

    def update(self, sample):
        self.t +=1
        #fa una media dei valori nell'arco temporale M
        if self.t <=self.M:
            self.reference += sample/self.M
            return 0
        else:
            #all'M+1 esima iterazione controlla se è presente un change della media
            self.reference = (self.reference * (self.t - 1) + sample) / self.t
            # se s+ = (x - x') - eps o s- = (x' - x) - eps dove x' è la media dei valori nell'arco temporale M
            s_plus= (sample - self.reference) - self.eps
            s_minus = -(sample - self.reference) - self.eps


            self.g_plus = max(0,self.g_plus + s_plus)
            self.g_minus = max(0,self.g_plus + s_minus)

            return self.g_plus > self.h or self.g_minus > self.h

    def reset(self):
        self.t=0
        self.g_minus=0
        self.g_plus=0

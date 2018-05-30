
class Fib:
    def __init__(self, max):
        self.max = max
    def __iter__(self):
        self.a = 0
        self.b = 1
        return self

    def __next__(self):
        fib = self.a
        if fib > self.max:
            raise StopIteration
        self.a, self.b = self.b, self.a + self.b
        return fib
'''
a=Fib(100)
for i in a:
    print(i)
for i in a:
    print(i)
'''

import torch
from torch.nn.utils.rnn import pad_sequence
a = torch.ones(25, 300)
b = torch.ones(22, 300)
c = torch.ones(15, 300)
print(pad_sequence([ b, a, c]).size())

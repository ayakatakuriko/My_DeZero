if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Function
import numpy as np
import math

from dezero.core_simple import Variable

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = gy * np.cos(x)
        return gx

def sin(x):
    return Sin()(x)

def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        t: Variable = (-1)**i * (x**(2 * i + 1) / math.factorial(2 * i + 1))
        y = y + t
        if abs(t.data) < threshold:
            break
    return y

if __name__=="__main__":
    x = Variable(np.array(np.pi/4))
    y = my_sin(x)
    y.backward()

    print(y.data)
    print(x.grad)

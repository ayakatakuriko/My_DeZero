if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import numpy as np

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.))
iters = 10

for i in range(iters):
    print(i, x)

    y: Variable = f(x)
    x.cleargrad()
    y.backward(create_graph=True)

    gx: Variable = x.grad
    x.cleargrad()
    gx.backward()
    gx2: Variable =x.grad
    x.data -= gx.data / gx2.data
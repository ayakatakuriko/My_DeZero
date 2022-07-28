if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
from dezero.utils import plot_dot_graph
import numpy as np

def shpare(x, y):
    z = x ** 2 + y ** 2
    return z

def matyas(x, y):
    z = 0.26 * (x ** 2 + y ** 2) - 0.48 * x * y
    return z

def goldstein(x, y):
    z = (1 + (x + y + 1)**2 * (19 - 14*x + 3*x**2 - 14*y + 6*x*y +3*y**2)) * \
        (30 + (2*x - 3*y)**2 * (18 - 32*x + 12*x**2 + 48*y - 36*x*y + 27*y**2))
    return z

if __name__=="__main__":
    x: Variable = Variable(np.array(1.0))
    y: Variable = Variable(np.array(1.0))
    #z: Variable = shpare(x, y)
    #z: Variable = matyas(x, y)
    z: Variable = goldstein(x, y)
    z.backward()
    print(x.grad, y.grad)

    x.name = "x"
    y.name = "y"
    z.name = "z"
    plot_dot_graph(z, verbose=False, to_file="goldtein.png")
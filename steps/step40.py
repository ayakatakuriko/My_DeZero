if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import dezero.functions as F
import numpy as np
from dezero.utils import plot_dot_graph

x0: Variable = Variable(np.array([[1, 2, 3]]))
x1: Variable = Variable(np.array([[10]]))
y: Variable = x0 + x1
y.backward()
print(y)
print(x1.grad)
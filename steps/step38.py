if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import dezero.functions as F
import numpy as np
from dezero.utils import plot_dot_graph

x: Variable = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y: Variable = F.reshape(x, (6, ))
y.backward(retain_grad=True)
print(x.grad)
print(x.reshape((3, 2)))
print(x.reshape(3, 2))

x1: Variable = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y1: Variable = F.transpose(x1)
y1.backward()
print(x1.grad)

x2: Variable = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
print(x2.T)
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import dezero.functions as F
import numpy as np
from dezero.utils import plot_dot_graph

x: Variable = Variable(np.array([[1, 2, 3, 4, 5, 6]]))
y: Variable = F.sum(x)
y.backward(retain_grad=True)
print(x.grad)

x1: Variable = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
y1: Variable = F.sum(x1, axis=0)
y1.backward()
print(x1.grad)

x2 = Variable(np.random.randn(2, 3, 4, 5))
y2 = x2.sum(keepdims=True)
print(y2.shape)
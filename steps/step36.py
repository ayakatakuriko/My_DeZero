if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import dezero.functions as F
import numpy as np
from dezero.utils import plot_dot_graph

x: Variable = Variable(np.array(2.0))
y: Variable = x ** 2
y.backward(create_graph=True)
gx: Variable = x.grad
x.cleargrad()

z: Variable = gx ** 3 + y
z.backward()
print(x.grad)
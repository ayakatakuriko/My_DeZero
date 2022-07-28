if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import dezero.functions as F
import numpy as np
from dezero.utils import plot_dot_graph

x:Variable = Variable(np.array(1.0))
y: Variable = F.tanh(x)
x.name = "x"
y.name = "y"
y.backward(create_graph=True)

iters: int = 0
for i in range(iters):
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

gx: Variable = x.grad
gx.name = "gx" + str(iters+1)
plot_dot_graph(gx, verbose=False, to_file="tanh.png")
if "__file__" in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dezero import Variable
import dezero.functions as F
import numpy as np

x: Variable = Variable(np.random.randn(2, 3))
W: Variable = Variable(np.random.randn(3, 4))
y: Variable = F.matmul(x, W)
y.backward()

print(x.grad.shape)
print(W.grad.shape)
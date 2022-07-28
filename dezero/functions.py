import numpy as np
from dezero.core import Function, Variable

class Sin(Function):
    def forward(self, x: np.array) -> np.array:
        y = np.sin(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * cos(x)
        return gx

def sin(x: Variable):
    return Sin()(x)

class Cos(Function):
    def forward(self, x: np.array) -> np.array:
        y = np.cos(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        x, = self.inputs
        gx = gy * -sin(x)
        return gx

def cos(x: Variable):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x: np.array) -> np.array:
        y = np.tanh(x)
        return y

    def backward(self, gy: Variable) -> Variable:
        y = self.outputs[0]()
        gx = gy * (1 - y * y)
        return gx

def tanh(x: Variable):
    return Tanh()(x)
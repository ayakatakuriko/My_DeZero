import numpy as np
import weakref
from heapq import *
import contextlib

from memory_profiler import profile

class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data        
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def backward(self, retain_grad=False):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = []
        seen_set = set()

        def add_func(f: Function):
            if f not in seen_set:
                heappush(funcs, (-f.generation, f))
                seen_set.add(f)

        def pop_func():
            _, f = heappop(funcs)
            return f

        add_func(self.creator)
        while funcs:
            f = pop_func()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx if x.grad is None else gx + x.grad
                if x.creator is not None:
                    add_func(x.creator)
            if not retain_grad:
                for y in f.outputs:
                    y().grad = None

    def cleargrad(self):
        self.grad = None

def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x

class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys, )
        outputs = [Variable(as_array(y)) for y in ys]

        if Config.enable_backprop:
            self.generation = max([x.generation for x in inputs])
            for output in outputs:
                output.set_creator(self)
            self.inputs = inputs
            self.outputs = [weakref.ref(output) for output in outputs]
        
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):                
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError

    def __lt__(self, other):
        return type(self).__name__ < type(other).__name__

class Config:
    enable_backprop = True

@contextlib.contextmanager
def using_config(name, value):
    old_value = getattr(Config, name)
    setattr(Config, name, value)
    try:
        yield
    finally:
        setattr(Config, name, old_value)

def no_grad():
    return using_config("enable_backprop", False)

class Add(Function):
    def forward(self, x1, x2):
        return x2 + x1

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

class Square(Function):
    def forward(self, x):
        return x ** 2
    
    def backward(self, gy):
        x = self.inputs[0].data
        return 2 * x *gy

def square(x):
    return Square()(x)

if __name__ == "__main__":
    @profile
    def test_backprop():
        with using_config("enable_backprop", True):
            x: Variable = Variable(np.ones((100, 100, 100)))
            y = square(square(square(x)))
            y.backward()
    
    @profile
    def test_wo_backprop():
        with no_grad():
            x: Variable = Variable(np.ones((100, 100, 100)))
            y = square(square(square(x)))

    test_backprop()
    test_wo_backprop()
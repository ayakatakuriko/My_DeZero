from turtle import forward
import numpy as np

class Variable:
    def __init__(self, data: np.ndarray) -> None:
        if data is not None:
            if not isinstance(data, np.ndarray):
                raise TypeError("{} is not supported".format(type(data)))
        self.data = data        
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)
        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs, )
            for x, gx in zip(f.inputs, gxs):
                x.grad = gx
                if x.creator is not None:
                    funcs.append(x.creator)

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
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, xs):                
        raise NotImplementedError()

    def backward(self, gys):
        raise NotImplementedError

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
    x1 = Variable(np.array(2))
    x2 = Variable(np.array(3))
    y = add(x1, x2)
    print(y.data)
    z = add(square(x1), square(x2))
    z.backward()
    print(z.data)
    print(x1.grad)
    print(x2.grad)
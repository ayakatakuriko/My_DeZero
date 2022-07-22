from step09 import *

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

def add(x0, x1):
    return Add()(x0, x1)

if __name__ == "__main__":
    x1 = Variable(np.array(2))
    x2 = Variable(np.array(3))
    y = add(x1, x2)
    print(y.data)
import random

e = 1e-7

from operations import Ops, Operation
# from typing import List, Set, Tuple

class Value:
    def __init__(self, value, children=(), op=lambda:None, label=""):
        self.ops = Ops()  # Shared Ops instance
        self.value = value
        self.grad = 0.0
        self.children:tuple[Value] = children
        self.op = op  # Store operation name for debugging
        
        self.label = label
        if isinstance(op, Operation):
            self.op_name = op.op_name
        else:
            self.op_name = ""

    def __repr__(self):
        return f"Value(value={self.value}, grad={self.grad}, op={self.op_name})"

    def build_graph(self, v:"Value", visited:set, graph:list):
        if v not in visited:
            visited.add(v)
            for child in v.children:
                self.build_graph(child, visited, graph)
            graph.append(v)

    def backward(self):
        """Perform backpropagation using topological sorting."""
        graph:list[Value] = []
        visited = set()
        self.build_graph(self, visited, graph)
        
        self.grad = 1.0
        for v in reversed(graph):
            if isinstance(v.op, Operation):
                grads = v.op.backward(v.grad)  # Compute gradients
                for child, grad in zip(v.children, grads):
                    child.grad += grad  # Accumulate gradients

    # ---- Generic Method to Apply an Operation ----
    def __apply_op(self, op, other=None):
        """Applies an operation from Ops with one or two inputs."""
        if other is None:
            inputs = [self.value,]
            out_chilren = (self,)

        elif isinstance(other, Value):
            inputs = [self.value, other.value]
            out_chilren = (self, other)

        else:
            other =Value(other)
            inputs = [self.value, other.value]
            out_chilren = (self, other)

        result = op(*inputs)
        
        out = Value(result, out_chilren, op)
        return out

    # ---- Overloaded Operators ----
    def __add__(self, other): return self.__apply_op(self.ops.add, other)
    def __sub__(self, other): return self.__apply_op(self.ops.sub, other)
    def __mul__(self, other): return self.__apply_op(self.ops.mul, other)
    def __truediv__(self, other): return self.__apply_op(self.ops.div, other)
    def __pow__(self, other): return self.__apply_op(self.ops.pow, other)

    # ---- Overloaded Reverse Operators ----
    def __radd__(self, other): return self.__add__(other)
    def __rmul__(self, other): return self.__mul__(other)
    
    def __rsub__(self, other): return Value(other).__sub__(self)
    def __rtruediv__(self, other): return Value(other).__truediv__(self)
    def __rpow__(self, other): return Value(other).__pow__(self)

    # ---- Activation Functions (Unary Ops) ----
    def exp(self): return self.__apply_op(self.ops.exp)
    def log(self): return self.__apply_op(self.ops.log)
    def sigmoid(self): return self.__apply_op(self.ops.sigmoid)
    def tanh(self): return self.__apply_op(self.ops.tanh)
    def relu(self): return self.__apply_op(self.ops.relu)


class Module:

    # def zero_grad(self):
    #     for p in self.parameters():
    #         p.grad = 0

    def parameters(self):
        return []

    def __repr__(self):
        return f"Module({self.__class__.__name__})"
    
    def __call__(self, *args):
        return self.forward(*args)
    
    def forward(self, x:list[Value]):
        raise NotImplementedError

class Neuron(Module):
  
    def __init__(self, nin):
        self.nin = nin
        self.w = []
        for _ in range(nin+1):
            # wi = Value(1.0)
            wi = Value(random.uniform(-1, 1))
            self.w.append(wi)

    def forward(self, x:list[Value]):
        # out = w0 + w1*x1 + w2*x2
        out = self.w[0]
        for xi, wi in zip(x, self.w[1:]):
            out = out + xi * wi
        return out

    def __repr__(self):
        return f"Neuron(nin={self.nin})"

    def parameters(self):
        return self.w

class Layer(Module):
    def __init__(self, nin, nout):
        self.nin = nin
        self.nout = nout
        self.neurons:list[Neuron] = []
        for _ in range(nout):
            self.neurons.append(Neuron(nin))

    def forward(self, x):
        out = []
        for neuron in self.neurons:
            out.append(neuron(x))
        return out

    def parameters(self):
        params = []
        for neuron in self.neurons:
            params.extend(neuron.parameters())
        return params

    def __repr__(self):
        return f"Layer(nin={self.nin}, nout={self.nout})"
    

class Sequential(Module):
    def __init__(self, *modules):
        self.modules:list[Module] = list(modules)

    def forward(self, x):
        for module in self.modules:
            x = module(x)
        return x

    def parameters(self):
        params = []
        for module in self.modules:
            params.extend(module.parameters())
        return params

    def __repr__(self):
        return f"Sequential({self.modules})"


class Activation(Module):
    def __init__(self):
        self.act = self.__class__.__name__.lower()

    def forward(self, x):
        out = [None] * len(x)
        for i in range(len(x)):
            out[i] = getattr(x[i], self.act)()
        return out

    def __repr__(self):
        return f"Activation({self.act})"

class Sigmoid(Activation):
    pass

class Relu(Activation):
    pass

class Tanh(Activation):
    pass


class BCELoss(Module):
    def forward(self, y: list[Value], p: list[Value]):
        totalloss = 0
        m = len(y)
        for yi, pi in zip(y, p):
            loss = -1 * (yi * (pi + e).log() + (1 - yi) * (1 - pi + e).log())
            totalloss += loss
        avgloss = totalloss / m
        return avgloss

class SGD:
    def __init__(self, params: list[Value], lr=0.01):
        self.lr = lr
        self.params = params

    def zero_grad(self):
        for p in self.params:
            p.grad = 0

    def step(self):
        for p in self.params:
            p.value = p.value - self.lr * p.grad

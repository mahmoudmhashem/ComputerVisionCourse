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
            other = Value(other)
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

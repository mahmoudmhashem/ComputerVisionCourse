import math

class Operation:
    def __init__(self):
        self.op_name = self.__class__.__name__.lower()

    def _store(self, *args):
        self.inputs = args
    
    def forward(self, *inputs):
        raise NotImplementedError
    
    def backward(self, output_grad):
        raise NotImplementedError

    def __call__(self, *args):
        return self.forward(*args)  # Call forward when the instance is used

class Add(Operation):
    def forward(self, a, b):
        self._store(a, b)
        result = a + b  # Perform addition
        return result
    
    def backward(self, output_grad):
        grad_a = output_grad # d(out)/da = 1
        grad_b = output_grad # d(out)/db = 1
        return grad_a, grad_b

class Multiply(Operation):
    def forward(self, a, b):
        self._store(a, b)
        result = a * b  # Perform multiplication
        return result

    def backward(self, output_grad):
        a, b = self.inputs
        grad_a = output_grad * b # d(out)/da = b
        grad_b = output_grad * a  # d(out)/db = a
        return grad_a, grad_b

class Exp(Operation):
    def forward(self, a):
        self._store(a)
        result = math.exp(a)  # e^a
        self.result = result
        return result

    def backward(self, output_grad):
        grad_a =  output_grad * self.result  # d(e^a)/da = e^a
        return grad_a, # retun a tuple

class Sigmoid(Operation):
    def forward(self, a):
        self._store(a)
        result = 1 / (1 + math.exp(-a))  # Sigmoid formula
        self.result = result  # Store for use in backward
        return result

    def backward(self, output_grad):
        grad_a = output_grad * self.result * (1 - self.result)  # d(sigmoid)/da = sigmoid(a) * (1 - sigmoid(a))
        return grad_a,

class Tanh(Operation):
    def forward(self, a):
        self._store(a)
        result = math.tanh(a)  # tanh formula
        self.result = result  # Store for use in backward
        return result

    def backward(self, output_grad):
        grad_a = output_grad * (1 - self.result ** 2)  # d(tanh)/da = 1 - tanh^2(a)
        return grad_a,

class Relu(Operation):
    def forward(self, a):
        self._store(a)
        result = max(0, a)  # ReLU formula
        return result

    def backward(self, output_grad):
        grad_a = output_grad * (1 if self.inputs[0] > 0 else 0)  # d(ReLU)/da = 1 if a > 0 else 0
        return grad_a,

class Log(Operation):
    def forward(self, a):
        self._store(a)
        result = math.log(a)  # Natural log
        return result

    def backward(self, output_grad):
        grad_a = output_grad / self.inputs[0]  # d(ln(a))/da = 1/a
        return grad_a,

class Divide(Operation):
    def forward(self, a, b):
        self._store(a, b)
        result = a / b  # Division formula
        return result

    def backward(self, output_grad):
        a, b = self.inputs
        grad_a = output_grad / b  # d(a / b) / da = 1 / b
        grad_b = output_grad * -a / (b ** 2)  # d(a / b) / db = -a / b^2
        return grad_a, grad_b

class Subtract(Operation):
    def forward(self, a, b):
        self._store(a, b)
        result = a - b
        return result

    def backward(self, output_grad):
        grad_a = output_grad # d(out)/da = 1
        grad_b = -output_grad  # d(out)/db = -1
        return grad_a, grad_b

class Power(Operation):
    def forward(self, a, b):
        self._store(a, b)
        result = a ** b  # a^b
        return result

    def backward(self, output_grad):
        a, b = self.inputs
        grad_a = output_grad * b * (a ** (b - 1)) if a != 0 else 0  # d(a^b)/da = b * a^(b-1)
        grad_b = output_grad * (a ** b) * math.log(a) if a > 0 else 0  # d(a^b)/db = a^b * ln(a)
        return grad_a, grad_b

class Ops: # Class to hold all operations
    def __init__(self):
        self.add = Add()
        self.sub = Subtract()
        self.mul = Multiply()
        self.div = Divide()
        self.exp = Exp()
        self.log = Log()
        self.pow = Power()
        self.sigmoid = Sigmoid()
        self.tanh = Tanh()
        self.relu = Relu()

def test_operations():
    ops = Ops()

    # Test Addition
    a, b = 3, 5
    result = ops.add(a, b)
    grads = ops.add.backward(1)
    print(f"Add({a}, {b}) = {result}, Backward: {grads}")

    # Test Subtraction
    result = ops.sub(a, b)
    grads = ops.sub.backward(1)
    print(f"Sub({a}, {b}) = {result}, Backward: {grads}")

    # Test Multiplication
    result = ops.mul(a, b)
    grads = ops.mul.backward(1)
    print(f"Mul({a}, {b}) = {result}, Backward: {grads}")

    # Test Division
    result = ops.div(a, b)
    grads = ops.div.backward(1)
    print(f"Div({a}, {b}) = {result}, Backward: {grads}")

    # Test Exponentiation
    result = ops.exp(a)
    grads = ops.exp.backward(1)
    print(f"Exp({a}) = {result}, Backward: {grads}")

    # Test Logarithm
    result = ops.log(b)
    grads = ops.log.backward(1)
    print(f"Log({b}) = {result}, Backward: {grads}")

    # Test Power
    base, exponent = 2, 3
    result = ops.pow(base, exponent)
    grads = ops.pow.backward(1)
    print(f"Pow({base}, {exponent}) = {result}, Backward: {grads}")

    # Test Sigmoid
    result = ops.sigmoid(0)
    grads = ops.sigmoid.backward(1)
    print(f"Sigmoid({a}) = {result}, Backward: {grads}")

    # Test Tanh
    result = ops.tanh(a)
    grads = ops.tanh.backward(1)
    print(f"Tanh({a}) = {result}, Backward: {grads}")

    # Test ReLU
    result = ops.relu(a)
    grads = ops.relu.backward(1)
    print(f"ReLU({a}) = {result}, Backward: {grads}")

    # Test ReLU with negative input
    neg_a = -3
    result = ops.relu(neg_a)
    grads = ops.relu.backward(1)
    print(f"ReLU({neg_a}) = {result}, Backward: {grads}")

# # Run the tests
# test_operations()

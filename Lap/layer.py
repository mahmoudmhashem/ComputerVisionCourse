from NN_from_scratch import Value
import random

e = 1e-7

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


def main():
    model = Sequential(
        Layer(3, 4),
        Relu(),
        Layer(4, 4),
        Relu(),
        Layer(4, 1),
        Sigmoid(),
    )

    criterion = BCELoss()
    optimizer = SGD(model.parameters(), lr=1)



    xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ]

    ys = [1.0, 0.0, 0.0, 1.0] # desired targets

    for k in range(100):
    
        # forward pass
        ypreds = []
        for x, y in zip(xs, ys):
            ypred = model(x)[0]
            ypreds.append(ypred)

        loss:Value = criterion(ys, ypreds)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Loss: {loss}")

    print()
    print("Predictions:")

    for ypred in ypreds:
        print(ypred)


if __name__ == "__main__":
    main()
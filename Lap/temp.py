from NN_from_scratch import Value
import random

class Neuron:
  
  def __init__(self, nin):
    self.w = [Value(1) for _ in range(nin)]
    self.b = Value(1)
  
  def __call__(self, x):
    # w * x + b
    act = sum((wi*xi for wi, xi in zip(self.w, x)), self.b)
    out = act.relu()
    return out
  
  def parameters(self):
    return self.w + [self.b]

class Layer:
  
  def __init__(self, nin, nout):
    self.neurons = [Neuron(nin) for _ in range(nout)]
  
  def __call__(self, x):
    outs = [n(x) for n in self.neurons]
    return outs[0] if len(outs) == 1 else outs
  
  def parameters(self):
    return [p for neuron in self.neurons for p in neuron.parameters()]

class MLP:
  
  def __init__(self, nin, nouts):
    sz = [nin] + nouts
    self.layers = [Layer(sz[i], sz[i+1]) for i in range(len(nouts))]
  
  def __call__(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
  
  def parameters(self):
    return [p for layer in self.layers for p in layer.parameters()]
  

x= [Value(1.58), Value(8.0), Value(738.0)]
n = MLP(3, [4, 4, 1])
o = n(x)

o.backward()

print(o)

print(x[0].grad, x[1].grad, x[2].grad, sep="\n")
from __future__ import annotations
from visualization import draw_dot
from NN_from_scratch import Value

# # inputs x1,x2
# x1 = Value(2.0, label='x1')
# x2 = Value(0.0, label='x2')

# # weights w1,w2
# w1 = Value(-3.0, label='w1')
# w2 = Value(1.0, label='w2')

# # bias of the neuron
# b = Value(6.8813735870195432, label='b')

# # x1*w1
# x1w1 = x1*w1; x1w1.label = 'x1*w1'
# # x2*w2
# x2w2 = x2*w2; x2w2.label = 'x2*w2'
# # x1*w1 + x2*w2
# x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
# # w1x1 + w2x2 + b
# n = x1w1x2w2 + b; n.label = 'n'

# o = n.tanh(); o.label = 'o'

# o.backward()

# print(f"x1.grad: {x1.grad}")
# print(f"w1.grad: {w1.grad}")
# print(f"x2.grad: {x2.grad}")
# print(f"w2.grad: {w2.grad}")



# inputs x1,x2
x1 = Value(2.0, label='x1')
x2 = Value(0.0, label='x2')
# weights w1,w2
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
# bias of the neuron
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = 'x1*w1'
x2w2 = x2*w2; x2w2.label = 'x2*w2'
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
# ----
e = (2*n).exp()
o = (e - 1) / (e + 1)
# ----
o.label = 'o'
o.backward()

draw_dot(o)
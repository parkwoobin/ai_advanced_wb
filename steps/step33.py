if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable

def f(x):
    y = x ** 4 - 2 * x ** 2
    return y

x = Variable(np.array(2.0))
y = f(x)
y.backward(create_graph=True)
print(x.grad)  # x.grad는 0.0이 아닌 0.0에 가까운 값이 나옴

gx = x.grad
x.cleargrad()
gx.backward()
print(x.grad)  # x.grad는 0.0이 아닌 0.0에 가까운 값이 나옴
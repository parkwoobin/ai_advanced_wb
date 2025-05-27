if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
import dezero.functions as F

x = Variable(np.array([[1, 2, 3], [4, 5, 6]]))
c = Variable(np.array([[10, 20, 30], [40, 50, 60]]))
t = x + c
y = F.sum(t)

y.backward(retain_grad=True)
print("y.grad : ",y.grad)
print("y : ",y.data)
print("t.grad : ",t.grad)
print("t : ",t.data)
print("x.grad : ",x.grad)
print("x : ",x.data)
print("c.grad : ",c.grad)
print("c : ",c.data)
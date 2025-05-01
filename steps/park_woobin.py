if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from dezero.core_simple import Variable

def f(x):
    y = x ** 4 + 4 * x ** 2 + 2025
    return y

def gx2(x):
    return 12 * x ** 2 + 8

x = Variable(np.array(100.0))  # 초깃값을 float으로 설정
iters = 30  # 반복 횟수

for i in range(iters):
    print(i, x)

    y = f(x)
    x.cleargrad()
    y.backward()

    x.data -= x.grad / gx2(x.data)  # 뉴턴 방법 적용

# 약 15회 반복 후 x = 0에 수렴함
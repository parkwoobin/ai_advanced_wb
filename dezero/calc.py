import os
import sys

# dezero 디렉토리를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


from dezero.core import Variable
from functions import square, sin, cos, exp, log
import numpy as np  # numpy 추가

def run_calc():
    x = Variable(np.array(3.0), name='x')  # float 데이터를 numpy 배열로 변환

    # 다양한 수식 실험
    y1 = square(x)           # x^2
    y2 = sin(x) + cos(x)     # sin(x) + cos(x)
    y3 = exp(x) * log(x)     # e^x * log(x)

    # 역전파 계산
    y1.backward(retain_grad=True)
    print("y = x^2:")
    print("값:", y1.data, "미분:", x.grad)

    x.cleargrad()
    y2.backward(retain_grad=True)
    print("y = sin(x) + cos(x):")
    print("값:", y2.data, "미분:", x.grad)

    x.cleargrad()
    y3.backward()
    print("y = exp(x) * log(x):")
    print("값:", y3.data, "미분:", x.grad)


if __name__ == '__main__':
    run_calc()
import os
import sys

# dezero 디렉토리를 sys.path에 추가
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero.core import Variable
from functions import square
import numpy as np  # numpy 추가

def simple_demo():
    x = Variable(np.array(2.0))  # float 데이터를 numpy 배열로 변환
    a = square(x)
    b = square(a)
    y = square(b)

    y.backward()
    print("=== 간단한 Demo ===")
    print("최종 출력값 y:", y.data)
    print("x에 대한 미분 dy/dx:", x.grad)


if __name__ == '__main__':
    simple_demo()
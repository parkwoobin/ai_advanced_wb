import numpy as np
from core_simple import Variable
from utils import get_dot_graph

x0 = Variable(np.array(1.0))
x1 = Variable(np.array(1.0))
y= x0+ x1 # 어떤 계산

# 변수 이름 지정
x0.name = 'x0'
x1.name = 'x1'
y.name = 'y'

txt = get_dot_graph(y, verbose=False)
print(txt)
# dot 파일로 저장
with open('sample.dot', 'w') as o:
    o.write(txt)
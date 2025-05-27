import numpy as np
from dezero.core import Function, Variable
# -------------------
# 기본 연산 함수들
# -------------------

class Square(Function):
    def forward(self, x):
        return x ** 2

    def backward(self, gy):
        x, = self.inputs
        return gy * 2 * x

def square(x):
    return Square()(x)


class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        self.y = y  # 저장해두기
        return y

    def backward(self, gy):
        return gy * self.y

def exp(x):
    return Exp()(x)


class Log(Function):
    def forward(self, x):
        return np.log(x)

    def backward(self, gy):
        x, = self.inputs
        return gy / x

def log(x):
    return Log()(x)

# -------------------
# 삼각 함수
# -------------------

class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        return gy * cos(x)

def sin(x):
    return Sin()(x)


class Cos(Function):
    def forward(self, x):
        y = np.cos(x)
        return y

    def backward(self, gy):
        x, = self.inputs
        return gy * -sin(x)

def cos(x):
    return Cos()(x)

#--37
class Sum(Function):
    def forward(self, x):
        self.keepdims = getattr(self, 'keepdims', False)
        return x.sum(axis=None, keepdims=self.keepdims)

    def backward(self, gy):
        x = self.inputs[0]
        gx = gy * np.ones_like(x.data)
        return gx

#--38
class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape  # 입력 데이터의 원래 형태 저장
        return x.reshape(self.shape)

    def backward(self, gy):
        return Variable(gy.data.reshape(self.x_shape))  # gy.data를 reshape하고 Variable로 감싸기

def reshape(x, shape):
    return Reshape(shape)(x)

class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        return x.transpose(self.axes)

    def backward(self, gy):
        if self.axes is None:
            return Variable(gy.data.transpose())  # gy.data를 transpose하고 Variable로 감싸기
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return Variable(gy.data.transpose(inv_axes))  # gy.data를 transpose하고 Variable로 감싸기

def transpose(x, axes=None):
    return Transpose(axes)(x)
#--

def sum(x, keepdims=False):
    return Sum()(x)

class Mul(Function):
    def forward(self, x0, x1):
        return x0 * x1

    def backward(self, gy):
        x0, x1 = self.inputs
        return gy * x1, gy * x0

def mul(x0, x1):
    return Mul()(x0, x1)

class Add(Function):
    def forward(self, x0, x1):
        return x0 + x1

    def backward(self, gy):
        return gy, gy

def add(x0, x1):
    return Add()(x0, x1)

class Neg(Function):
    def forward(self, x):
        return -x

    def backward(self, gy):
        return -gy

def neg(x):
    return Neg()(x)

class Div(Function):
    def forward(self, x0, x1):
        return x0 / x1

    def backward(self, gy):
        x0, x1 = self.inputs
        gx0 = gy / x1
        gx1 = gy * (-x0 / (x1 ** 2))
        return gx0, gx1

def div(x0, x1):
    return Div()(x0, x1)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        self.y = y  # 저장해두기
        return y

    def backward(self, gy):
        return gy * (1 - self.y ** 2)

def tanh(x):
    return Tanh()(x)
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# sys.path 설정 (DeZero 모듈 사용 시 필요)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F


def get_function(name):
    """다양한 수식 함수 모드 제공"""
    if name == "square3":
        def square3(x):
            return F.square(F.square(F.square(x)))
        return square3
    elif name == "sin":
        return F.sin
    elif name == "exp":
        return F.exp
    elif name == "tanh":
        return F.tanh
    else:
        raise ValueError("지원하지 않는 함수입니다.")


def visualize_function_and_gradient(func, x_input, func_name):
    """함수와 도함수를 시각화"""
    x = Variable(np.array(x_input))
    y = func(x)
    y.backward()

    print(f"입력값 x: {float(x.data)}")
    print(f"계산값 y: {float(y.data)}")
    print(f"미분값 dy/dx: {float(x.grad.data)}")

    xs = np.linspace(-3, 3, 200)
    ys = []
    grads = []

    for val in xs:
        x_var = Variable(np.array(val))
        y_val = func(x_var)
        y_val.backward()
        ys.append(float(y_val.data))
        grads.append(float(x_var.grad.data) if x_var.grad is not None else 0.0)

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, label='f(x)', color='blue')
    plt.plot(xs, grads, label="df/dx", linestyle='--', color='red')
    plt.scatter([float(x.data)], [float(y.data)], color='black', s=50, label=f"x = {float(x.data)}")
    plt.title("Function and Gradient Visualization")
    plt.xlabel("x")
    plt.ylabel("f(x) / df/dx")
    plt.grid()
    plt.legend()

    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    filename = os.path.join(output_dir, f"graph_{func_name}_x{float(x.data)}.png")
    plt.savefig(filename)
    print(f"그래프가 저장되었습니다: {filename}")
    plt.show()

def simulate_sgd_step():
    """SGD 학습 시뮬레이션"""
    print("SGD 학습 시뮬레이션을 시작합니다.")

    # 임의의 손실 함수 정의 (예: y = (x - 3)^2)
    x = Variable(np.array(0.0))  # 초기값
    learning_rate = 0.1

    for i in range(100):  # 100번 반복
        y = F.square(x - 3)  # 손실 함수
        x.cleargrad()
        y.backward()  # 역전파
        x.data -= learning_rate * x.grad.data  # SGD 업데이트

        if i % 10 == 0:  # 10번마다 출력
            print(f"Step {i}: x = {x.data}, y = {y.data}")

    print(f"최종 결과: x = {x.data}, y = {y.data}")

def main():
    print("[1] 함수 시각화")
    print("[2] SGD 학습 시뮬레이션")
    mode = input("모드를 선택하세요 (1/2): ")

    if mode == "1":
        print("수식 모드: square3, sin, exp, tanh 중 선택")
        func_name = input("함수 이름 입력: ")
        x_val = float(input("x 값 입력: "))
        func = get_function(func_name)
        visualize_function_and_gradient(func, x_val, func_name)

    elif mode == "2":
        simulate_sgd_step()

    else:
        print("잘못된 입력입니다.")


if __name__ == '__main__':
    main()

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from PyQt5.QtWidgets import (QApplication, QWidget, QLabel, QVBoxLayout,
                             QPushButton, QLineEdit, QMessageBox, QTextEdit, QComboBox)

# sys.path 설정 (DeZero 모듈 사용 시 필요)
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from dezero import Variable
import dezero.functions as F


def get_function(name):
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


def simulate_sgd_step(x_val):
    log = []
    x = Variable(np.array(x_val))
    learning_rate = 0.1
    for i in range(100):
        y = F.square(x - 3)
        x.cleargrad()
        y.backward()
        x.data -= learning_rate * x.grad.data
        if i % 10 == 0:
            log.append(f"Step {i}: x = {x.data:.6f}, y = {y.data:.6f}")
    log.append(f"최종 결과: x = {x.data:.6f}, y = {y.data:.6f}")
    return "\n".join(log)


def simulate_newton_method(x_val):
    def f(x):
        return x ** 4 + 4 * x ** 2 + 2025

    def gx2(x):
        return 12 * x ** 2 + 8

    log = []
    x = Variable(np.array(x_val))
    for i in range(30):
        y = f(x)
        x.cleargrad()
        y.backward()
        x.data -= x.grad.data / gx2(x.data)
        log.append(f"{i}회차: x = {float(x.data):.6f}, y = {float(y.data):.6f}, grad = {float(x.grad.data):.6f}")
    return "\n".join(log)


def visualize_function_and_gradient(func, x_input, func_name):
    x = Variable(np.array(x_input))
    y = func(x)
    y.backward()

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
    plt.savefig(filename)  # 그래프 저장
    plt.show()  # 그래프 화면에 표시
    return filename

class OptimizerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle("최적화 시뮬레이터")
        self.setGeometry(100, 100, 500, 400)

        layout = QVBoxLayout()

        self.mode_label = QLabel("모드 선택:")
        layout.addWidget(self.mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["function", "SGD", "Newton"])
        self.mode_combo.currentTextChanged.connect(self.update_ui_state)
        layout.addWidget(self.mode_combo)

        self.func_label = QLabel("함수 선택:")
        layout.addWidget(self.func_label)

        self.func_combo = QComboBox()
        self.func_combo.addItems(["square3", "sin", "exp", "tanh"])
        layout.addWidget(self.func_combo)

        self.x_label = QLabel("초기 x 값 입력:")
        layout.addWidget(self.x_label)

        self.x_input = QLineEdit("1.0")
        layout.addWidget(self.x_input)

        self.run_button = QPushButton("실행")
        self.run_button.clicked.connect(self.run_simulation)
        layout.addWidget(self.run_button)

        self.result_box = QTextEdit()
        self.result_box.setReadOnly(True)
        layout.addWidget(self.result_box)

        self.setLayout(layout)
        self.update_ui_state()

    def update_ui_state(self):
        mode = self.mode_combo.currentText()
        is_function_mode = (mode == "function")
        self.func_label.setEnabled(is_function_mode)
        self.func_combo.setEnabled(is_function_mode)

    def run_simulation(self):
        try:
            x_val = float(self.x_input.text())
        except ValueError:
            QMessageBox.warning(self, "입력 오류", "유효한 숫자를 입력해주세요.")
            return

        mode = self.mode_combo.currentText()

        if mode == "function":
            func_name = self.func_combo.currentText()
            try:
                func = get_function(func_name)
                filename = visualize_function_and_gradient(func, x_val, func_name)
                self.result_box.append(f"그래프가 시각화되어 저장되었습니다: {filename}")
            except Exception as e:
                self.result_box.append(str(e))
        elif mode == "SGD":
            result = simulate_sgd_step(x_val)
            self.result_box.append(result)
        elif mode == "Newton":
            result = simulate_newton_method(x_val)
            self.result_box.append(result)
        else:
            self.result_box.append("지원하지 않는 모드입니다.")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = OptimizerApp()
    ex.show()
    sys.exit(app.exec_())
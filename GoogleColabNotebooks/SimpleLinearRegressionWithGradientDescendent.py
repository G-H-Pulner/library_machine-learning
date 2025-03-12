import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:

    def __init__(self, alpha=0.01, iter=200, w=0, b=0):
        self.alpha = alpha
        self.iter = iter
        self.w = w
        self.b = b

    def fit(self, X, y):
        alpha = self.alpha
        n = x.shape[0]
        for _ in range(self.iter):
            px = self.w * X + self.b # é o (p(x1), p(x2), ..., p(xn))
            dw = np.dot(x, px - y) / n
            db = np.sum(px - y) / n
            # gradiente descendente
            self.w -= alpha * dw
            self.b -= alpha * db

    def predict(self, x):
        return self.w * x + self.b
    
# criando o conjunto de dados
def make_line(a, b):
    # cria pontos próximos da reta ax+b
    p, q = -1, 1
    def line(x):
        erro = p + (q - p) * np.random.random()
        return a * x + b + erro
    return line

a, b = 0.35, 0.48
line = make_line(a, b)

x = np.linspace(0, 20, 50)
y = [line(xi) for xi in x]

plt.scatter(x, y)
plt.show()

model = LinearRegression(alpha=0.001, iter=1000)
model.fit(x, y)

w = model.w
b = model.b

def reta(x):
    return w * x + b

y_pred = [reta(xi) for xi in x]

plt.scatter(x, y)
plt.plot(x, y_pred, color='red')
plt.show()
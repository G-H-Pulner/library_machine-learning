import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

class SimpleLinearRegression:
    """
    Versão cálculo 1: usando máximos e mínimos

    Parâmetros
    ----------
    x: variável dependente, a que explica y,
       feature, atributo, característica
    y: variável independente, a ser predita
    """

    def linear_regression(self, x, y):
        self.x = np.array(x)
        self.y = np.array(y)
        A = [
            [sum(x ** 2), sum(x)],
            [sum(x), len(x)]
        ]
        B = [sum(x * y),sum(y)]
        return np.linalg.solve(A, B)
    
    def split(self, x, y, test_size=0.3):
        self.x = x
        self.y = y
        dados = set(zip(x, y))
        k = int(test_size * 10)
        test = set(random.sample(list(dados), k))
        train = dados - test
        return *list(zip(*train)), *list(zip(*test))

    def fit(self, x, y):
        self.x = x
        self.y = y
        w, b = SimpleLinearRegression.linear_regression(x, y)
        self.w = w
        self.b = b
        self.line = SimpleLinearRegression.best_line(w, b)

    def predict(self, x):
        return self.line(x)
    
    def best_line(self, w, b):
        self.w = w
        self.b = b
        def line(self, x):
            self.x = x
            return w * x + b
        return line
    
    def RMSE(self, x, y):
        # Root Mean Squared Error
        # uma espécie de desvio padrão
        return np.sqrt(np.sum([(yi - self.predict(xi))**2 for xi, yi in zip(x, y)]) / len(x))
    
    def plot(self, name='output.png', points=200):
        t = np.linspace(min(self.x), max(self.x), points)
        ft = [self.predict(ti) for ti in t]
        sns.scatterplot(x=self.x, y=self.y)
        sns.lineplot(x=t, y=ft, color='red')
        plt.savefig(name)
        plt.show()

class SimpleLinearRegressionWithGradientDescendent:
    
    def __init__(self, alpha=0.01, iter=200, w=0, b=0):
        self.alpha = alpha
        self.iter = iter
        self.w = w
        self.b = b

    def fit(self, X, y):
        alpha = self.alpha
        n = X.shape[0]
        for _ in range(self.iter):
            px = self.w * X + self.b # é o (p(x1), p(x2), ..., p(xn))
            dw = np.dot(X, px - y) / n
            db = np.sum(px - y) / n
            # gradiente descendente
            self.w -= alpha * dw
            self.b -= alpha * db

    def predict(self, x):
        return self.w * x + self.b
    
    def best_line(self, w, b):
        self.w = w
        self.b = b
        def line(self, x):
            self.x = x
            return w * x + b
        return line

class LinearRegressionWithGradientDescendent:

    def __init__(self, alpha=0.01, iter=200):
        self.alpha = alpha # learning rate
        self.iter = iter # número de iterações
        self.w = None
        self.b = None

    def fit(self, X, y):
        n, m = X.shape
        if self.w is None:
            self.w = np.zeros(m)
        if self.b is None:
            self.b = 0

        for _ in range(self.iter):
            # y_hat = (p(x1), p(x2), ..., p(xm))
            y_hat = (self.w @ X.T) + self.b
            Y = y_hat - y
            # todas as derivadas parciais dJ/dwi
            dw = (Y @ X) / n # Y @ X multiplica Y pelas colunas de X
            # a derivada parcial dJ/db
            db = np.sum(Y) / n
            self.w -= self.alpha * dw
            self.b -= self.alpha * db

    def predict(self, x):
        return self.w @ x.T + self.b
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

class simple_regression:
    """
    Contem a regeressão linear utilizando 
    cálculo 1, usando máximos e mínimos, 
    juntamente com a versão utilizando o 
    gradiente descendente.

    ==========================================

    Versão cálculo 1: usando máximos e mínimos
    
    Parâmetros
    ----------
    x: variável dependente, a que explica y,
       feature, atributo, característica
    y: variável independente, a ser predita
    
    ==========================================

    Versão com gradiente descendente:

    Parâmetros
    ----------
    x: variável dependente, a que explica y,
       feature, atributo, característica
    y: variável independente, a ser predita
    """

    def split(self, x, y, test_size=0.3):
        self.x = x
        self.y = y
        dados = set(zip(x, y))
        k = int(test_size * 100)
        test = set(random.sample(list(dados), k))
        train = dados - test
        return *list(zip(*train)), *list(zip(*test))

    def fit(self, x, y):
        x = np.array(x)
        y = np.array(y)
        self.x = x
        self.y = y
        A = [
            [sum(x ** 2), sum(x)],
            [sum(x), len(x)]
        ]
        B = [sum(x * y),sum(y)]
        w, b = np.linalg.solve(A, B)
        self.w = w
        self.b = b

    def fit_gd(self, x, y, alpha=0.01, iter=200, w=0, b=0):
        x = np.array(x)
        y = np.array(y)
        self.x = x
        self.y = y
        self.alpha = alpha
        self.iter = iter
        self.w = w
        self.b = b
        n = x.shape[0]
        for _ in range(self.iter):
            px = self.w * x + self.b # é o (p(x1), p(x2), ..., p(xn))
            dw = np.dot(x, px - y) / n
            db = np.sum(px - y) / n
            # gradiente descendente
            self.w -= alpha * dw
            self.b -= alpha * db

    def predict(self, x):
        x = np.array(x)
        return self.w * x + self.b
    
    def RMSE(self, x, y):
        # Root Mean Squared Error
        # uma espécie de desvio padrão
        return np.sqrt(np.sum([(yi - self.predict(xi))**2 for xi, yi in zip(x, y)]) / len(x))
    
    def plot(self, name='output.png', points=200):
        t = np.linspace(min(self.x), max(self.x), points)
        ft = [self.predict(ti) for ti in t]
        sns.scatterplot(x=self.x, y=self.y, alpha = 0.5)
        sns.lineplot(x=t, y=ft, color='red')
        plt.savefig(name)
        plt.show()

class multiple_regression:

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
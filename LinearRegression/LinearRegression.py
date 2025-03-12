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

    def best_line(self, w, b):
        self.w = w
        self.b = b
        def line(self, x):
            self.x = x
            return w * x + b
        return line

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
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

def make_line(a, b):
    # cria pontos próximos da reta ax+b
    p, q = -1, 1
    def line(x):
        erro = p + (q - p) * np.random.random()
        return a * x + b + erro
    return line

a, b = 0.35, 0.48
line = make_line(a, b)

x = np.linspace(0, 20, 100)
y = np.array([line(xi) for xi in x])
#type(x)
#type(y)

#sns.scatterplot(x=x, y=y)
#sns.regplot(x=x,y=y)
#plt.show()

def custo(w, b):
    return np.sum([(yi - (w * xi + b))**2 for xi, yi in zip(x, y)])

#print(custo(0.35, 0.48))

# Make data.
w = np.linspace(-3, 3, 100)
b = np.linspace(-5, 5, 100)
w, b = np.meshgrid(w, b)
# create plots
z = np.array([custo(wi, bi) for wi, bi in zip(np.ravel(w), np.ravel(b))])
z = z.reshape(w.shape)

# Plot
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')

#ax.plot_surface(w, b, z)
#ax.set_xlabel('w Label')
#ax.set_ylabel('b Label')
#ax.set_zlabel('custo Label')
#plt.show()

# jeito ingênuo de fazer as coisas
def linear_regression(x, y):
    x = np.array(x)
    y = np.array(y)
    a = sum(x ** 2)
    b = sum(x)
    c = b
    d = len(x)
    A = [
        [a, b],
        [c, d]
    ]
    b1 = sum(x * y)
    b2 = sum(y)
    B = [
        b1,
        b2
    ]
    return np.linalg.solve(A, B)

# pontos usados w = 0.35, b = 0.48

w, b = linear_regression(x, y)
#print(w, b)
#print(custo(w, b))
#print(custo(0.35, 0.48))

def best_line(w, b):
    def line(x):
        return w * x + b
    return line

f = best_line(w, b)
t = np.linspace(min(x), max(x), 200)
ft = [f(ti) for ti in t]
#sns.scatterplot(x=x, y=y)
#sns.lineplot(x=t, y=ft, color='red')
#plt.show()

class LinearRegression:
    """
    Versão cálculo 1: usando máximos e mínimos

    Parâmetros
    ----------
    x: variável dependente, a que explica y,
       feature, atributo, característica
    y: variável independente, a ser predita
    """

    def fit(self, x, y):
        self.x = x
        self.y = y
        w, b = linear_regression(x, y)
        self.w = w
        self.b = b
        self.line = best_line(w, b)

    def predict(self, x):
        return self.line(x)

    def RMSE(self, x, y):
        # Root Mean Squared Error
        # uma espécie de desvio padrão
        return np.sqrt(np.sum([(yi - self.predict(xi))**2 for xi, yi in zip(x, y)]) / len(x))

    def plot(self, name='output.png'):
        t = np.linspace(min(self.x), max(self.x), 200)
        ft = [self.predict(ti) for ti in t]
        plt.scatter(self.x, self.y)
        plt.plot(t, ft, color='red')
        plt.savefig(name)
        plt.show()

#print(LinearRegression.__doc__)

linreg = LinearRegression()
linreg.fit(x, y)
linreg.w
linreg.b
linreg.line
linreg.predict(5.2398842938)
linreg.RMSE(x, y)
#linreg.plot()

# separar o conjunto de dados
def split(x, y, test_size=0.3):
    dados = set(zip(x, y))
    k = int(test_size * 10)
    test = set(random.sample(list(dados), k))
    train = dados - test
    return *list(zip(*train)), *list(zip(*test))

x_train, y_train, x_test, y_test = split(x, y) 
model = LinearRegression()
model.fit(x_train, y_train)
model.RMSE(x_train, y_train)
model.RMSE(x_test, y_test)

#model.plot()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.ensemble import RandomForestRegressor

class LinearRegression:

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

# testando: produto escalar
w = np.array([1,2,3])
X = np.array([
    [1,0,-1],
    [2,1,-2],
    [1,1,1],
])

# testando: multiplicando w pelas linhas de X
w @ X.T

# testando: esse produto de matrizes é o mesmo que o produto escalar
np.array([1,2,3]) @ np.array([-1,2,1])

df = pd.read_csv('DataSets/winequality-red.csv')

print(df.head()) # mostra as 5 primeiras linhas do arquivo
print(df.describe().T)

sns.histplot(data=df, x='total sulfur dioxide')
plt.show()

df_1 = (df - df.mean(axis=0)) / df.std(axis=0)
sns.histplot(data=df_1, x='total sulfur dioxide')
plt.show()   

print(df_1.head())
print(df_1.describe().T)

df_2 = df / df.max(axis=0)
sns.histplot(data=df_2, x='total sulfur dioxide')
plt.show()

print(df_2.head())
print(df_2.describe().T)

X = df.drop(columns=['quality'])
y = df['quality']
print(X.shape)

# random_state para garantir reprodutibilidade (é uma seed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
X_train.shape, X_test.shape

model = LinearRegression(alpha=0.0001, iter=10_000)
model.w = X.mean(axis=0)
model.b = y.mean(axis=0)
print(model.w)
print(model.b)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(y_pred)
print(model.w)
print(model.b)

def RMSE(y_pred, y_test):
    return np.sqrt(np.sum((y_pred - y_test)**2 / len(y_pred)))

print(RMSE(y_pred, y_test))

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X_test)
print(RMSE(y_pred, y_test))

model_sgd = SGDRegressor(alpha=0.01, max_iter=5_000, fit_intercept=False)
model_sgd.fit(X_train, y_train)
y_pred = model_sgd.predict(X_test)
print(RMSE(y_pred, y_test))

model_rfr = RandomForestRegressor()
model_rfr.fit(X_train, y_train)
y_pred = model_rfr.predict(X_test)
print(RMSE(y_pred, y_test))
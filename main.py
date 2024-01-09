import numpy as np
import scipy.stats as ss


class Convolution:
    def __init__(self, f, X, n):
        self.f = f
        self.X = X
        self.U = X[:, np.newaxis]
        self.dx = np.diff(X)[0]
        self.Y = self.f(self.U - self.X) * self.dx
        self.fX = self.f(self.X)
        self.n = n

    def pdf(self):
        return np.linalg.matrix_power(self.Y, self.n - 1) @ self.fX

    def cdf(self):
        density = self.pdf()
        return np.cumsum(density) * self.dx

    def quantile(self, p):
        _cdf = self.cdf()
        i = np.searchsorted(_cdf, p)
        return self.X[i - 1]


# %%time
X = np.linspace(0, 300, 5000)
g = Convolution(lambda x: ss.lognorm.pdf(x, s=1), X, 100)
g.quantile(0.95)

# %%time
# Quantile de la somme de deux lois lognormales standard
x = np.random.lognormal(size=(100, 1000000)).sum(axis=0)
np.quantile(x, 0.95)

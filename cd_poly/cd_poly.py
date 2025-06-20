from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import solve_triangular
import numpy as np


class CDPolynomial():

    def __init__(self, data, degree=2, eps=0.):
        """
        Initialize the CD polynomial based on data.
        Degree is the degree of the basis (half that of the moment matrix).
        We assume that the data points are represented by their coordinates in
        a basis - e.g. for a function space, the coordinates with respect to
        the Chebyshev basis. Eps is a regularization parameter which ensures
        that the moment matrix is nonsingular.
        """
        self.alg_deg = degree  # algebraic degree (highest deg of a monomial)
        _, self.harm_deg = data.shape  # harmonic degree (number of variables)

        poly = PolynomialFeatures(degree=degree).fit_transform(data)
        _, self.n_monomials = poly.shape
        self.moments = np.einsum('bi,bj->bij', poly, poly).mean(axis=0)
        self.moments += eps * np.eye(self.n_monomials, self.n_monomials)
        self.L = np.linalg.cholesky(self.moments)

    def __call__(self, z):
        """
        Evaluate the CD polynomial at an array of points
        (also represented by coords).
        """
        poly = PolynomialFeatures(degree=self.alg_deg).fit_transform(z)
        y = solve_triangular(self.L, poly.T, lower=True)  # (n_data, batch)
        z = solve_triangular(self.L.T, y, lower=False)  # (n_data, batch)
        return np.einsum('bi,ib->b', poly, z)
        # return np.einsum('bi,ib->b', poly,
        #                  np.linalg.solve(self.moments, poly.T))


def plot_contours(f, ax, **plot_kwargs):
    "Make a contour plot of a vectorized function on the given axes."

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(x, y)

    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    vals = f(points)
    Z = vals.reshape(X.shape)

    contours = ax.contour(X, Y, Z, alpha=0.9, **plot_kwargs)
    ax.clabel(contours)


def plot_level_set(f, alpha, ax):
    "Plot the alpha-level set of a vectorized function on the given axes."

    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    x = np.linspace(*xlim, 100)
    y = np.linspace(*ylim, 100)
    X, Y = np.meshgrid(x, y)

    points = np.stack([X.ravel(), Y.ravel()], axis=1)
    vals = f(points)
    Z = vals.reshape(X.shape)

    ax.contour(X, Y, Z, levels=[0., alpha], cmap='Blues', alpha=0.9)


def plot_func(f, ax, **plot_kwargs):
    "Plot a vectorized function on [-1, 1]."
    ts = np.linspace(-1, 1, 100)
    ax.plot(ts, f(ts), **plot_kwargs)


def sample_ball_unif(n_samples, rad, dim=2):
    "Sample uniformly from a Euclidean ball."
    x = np.random.randn(n_samples, dim)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    u = np.random.rand(n_samples, 1)
    r = rad * u ** (1.0 / dim)
    return r * x

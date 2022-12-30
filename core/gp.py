import gpytorch
import torch

from core.utils import uniform_samples


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, kernel, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def sample_gp_prior(kernel, bounds, num_points, jitter=1e-06):
    """
    Sample a random function from a GP prior with mean 0 and covariance specified by a kernel.
    :param kernel: a GPyTorch kernel.
    :param bounds: array of shape (2, num_dims).
    :param num_points: int.
    :param jitter: float.
    :return: Callable that takes in an array of shape (n, N) and returns an array of shape (n, 1).
    """
    points = uniform_samples(bounds=bounds, n_samples=num_points)
    cov = kernel(points).evaluate() + jitter * torch.eye(num_points)
    f_vals = torch.distributions.MultivariateNormal(
        torch.zeros(num_points, dtype=torch.double), cov
    ).sample()[:, None]

    L = torch.linalg.cholesky(cov)
    L_bs_f = torch.linalg.solve_triangular(L, f_vals, upper=False)
    LT_bs_L_bs_f = torch.linalg.solve_triangular(L.T, L_bs_f, upper=True)
    return lambda x: kernel(x, points).evaluate() @ LT_bs_L_bs_f

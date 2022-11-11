from abc import ABC, abstractmethod
from scipy.stats import norm
import torch


class Distribution(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, n_samples):
        pass

    @abstractmethod
    def variance(self):
        pass


class TruncNormDist(Distribution):
    def __init__(self, loc, scale, a, b):
        super().__init__()
        self.loc = loc
        self.scale = scale
        self.a = a
        self.b = b

    def sample(self, n_samples):
        return torch.tensor(
            truncnorm_samples(
                loc=self.loc, scale=self.scale, a=self.a, b=self.b, n_samples=n_samples
            ),
            dtype=torch.double,
        )

    def variance(self):
        return truncnorm_variance(loc=self.loc, scale=self.scale, a=self.a, b=self.b)


class UniformDist(Distribution):
    def __init__(self, a, b):
        super().__init__()
        self.a = a
        self.b = b

    def sample(self, n_samples):
        return uniform_samples_1d(a=self.a, b=self.b, n_samples=n_samples)

    def variance(self):
        return (1 / 12) * ((self.b - self.a) ** 2)


def truncnorm_transform(U, loc, scale, a, b):
    """
    :param U: samples from Uniform(0, 1).
    """
    alpha = (a - loc) / scale
    beta = (b - loc) / scale
    return (
        norm.ppf(norm.cdf(alpha) + U * (norm.cdf(beta) - norm.cdf(alpha))) * scale + loc
    )


def truncnorm_samples(loc, scale, a, b, n_samples):
    """
    Samples from a normal distribution with loc and scale but truncated to the range [a, b].
    """
    U = uniform_samples_1d(a=0.0, b=1.0, n_samples=n_samples)
    return truncnorm_transform(U=U, loc=loc, scale=scale, a=a, b=b)


def truncnorm_variance(loc, scale, a, b):
    alpha = (a - loc) / scale
    beta = (b - loc) / scale
    Z = norm.cdf(beta) - norm.cdf(alpha)

    A = (alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / Z
    B = ((norm.pdf(alpha) - norm.pdf(beta)) / Z) ** 2

    return (scale**2) * (1 + A - B)


def uniform_samples_1d(a, b, n_samples):
    return torch.rand(size=(n_samples,), dtype=torch.double) * (b - a) + a

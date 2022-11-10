from scipy.stats import norm, uniform


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
    U = uniform.rvs(size=n_samples)
    return truncnorm_transform(U=U, loc=loc, scale=scale, a=a, b=b)


def truncnorm_variance(loc, scale, a, b):
    alpha = (a - loc) / scale
    beta = (b - loc) / scale
    Z = norm.cdf(beta) - norm.cdf(alpha)

    A = (alpha * norm.pdf(alpha) - beta * norm.pdf(beta)) / Z
    B = ((norm.pdf(alpha) - norm.pdf(beta)) / Z) ** 2

    return (scale**2) * (1 + A - B)

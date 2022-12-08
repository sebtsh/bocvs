from abc import ABC, abstractmethod
from gpytorch.kernels import RFFKernel
import torch

from core.dists import get_opt_queries_and_vals
from core.utils import maximize_fn


def get_acquisition(acq_name):
    if acq_name == "ucb-cs":
        return UCB_PSQ_CS(beta=2.0)
    elif acq_name == "ucb":
        return UCB_PSQ(beta=2.0)
    elif acq_name == "ucb-naive":
        return UCB_PSQ_Naive(beta=2.0)
    elif acq_name == "ts":
        return TS_PSQ(n_features=1024)
    else:
        raise Exception("Incorrect acq_name passed to get_acquisition")


class Acquisition(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        pass


class TS_PSQ(Acquisition):
    """
    Adapted from https://botorch.org/tutorials/thompson_sampling.
    """

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        dims = bounds.shape[-1]

        # Thompson sampling via random Fourier features
        rff_k = RFFKernel(num_samples=self.n_features, num_dims=dims, ard_num_dims=dims)
        rff_k.lengthscale = gp.covar_module.base_kernel.lengthscale
        rff_k.randn_weights = rff_k.randn_weights.double()

        def featurize(X):
            return rff_k._featurize(X, normalize=True) * torch.sqrt(
                gp.covar_module.outputscale
            )

        Phi = featurize(train_X)
        s2 = gp.likelihood.noise
        PPinv = torch.linalg.inv(Phi @ Phi.T + s2 * torch.eye(Phi.shape[0]))
        mean = Phi.T @ PPinv @ train_y
        cov = torch.eye(Phi.shape[-1]) - Phi.T @ PPinv @ Phi
        L = torch.linalg.cholesky(cov)
        theta = mean + L @ torch.randn(size=mean.size(), dtype=torch.double)

        def f_sample(X):
            return featurize(X) @ theta

        skip_expectations = False
        for i, control_set in enumerate(control_sets):
            if len(control_set) == dims:
                skip_expectations = True
                ret_control_idx = i
                ret_query, _ = maximize_fn(f=f_sample, n_warmup=10000, bounds=bounds)
                ret_query = ret_query[None, :]
                break

        if not skip_expectations:
            opt_queries, opt_vals = get_opt_queries_and_vals(
                f=f_sample,
                control_sets=control_sets,
                random_sets=random_sets,
                all_dists_samples=all_dists_samples,
                bounds=bounds,
            )

            ret_control_idx = torch.argmax(opt_vals).item()
            ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query


class TS_PSQ_Naive(Acquisition):
    """
    Adapted from https://botorch.org/tutorials/thompson_sampling.
    """

    def __init__(self, n_features):
        super().__init__()
        self.n_features = n_features

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        dims = bounds.shape[-1]

        # Thompson sampling via random Fourier features
        rff_k = RFFKernel(num_samples=self.n_features, num_dims=dims, ard_num_dims=dims)
        rff_k.lengthscale = gp.covar_module.base_kernel.lengthscale
        rff_k.randn_weights = rff_k.randn_weights.double()

        def featurize(X):
            return rff_k._featurize(X, normalize=True) * torch.sqrt(
                gp.covar_module.outputscale
            )

        Phi = featurize(train_X)
        s2 = gp.likelihood.noise
        PPinv = torch.linalg.inv(Phi @ Phi.T + s2 * torch.eye(Phi.shape[0]))
        mean = Phi.T @ PPinv @ train_y
        cov = torch.eye(Phi.shape[-1]) - Phi.T @ PPinv @ Phi
        L = torch.linalg.cholesky(cov)
        theta = mean + L @ torch.randn(size=mean.size(), dtype=torch.double)

        def f_sample(X):
            return featurize(X) @ theta

        opt_queries, opt_vals = get_opt_queries_and_vals(
            f=f_sample,
            control_sets=control_sets,
            random_sets=random_sets,
            all_dists_samples=all_dists_samples,
            bounds=bounds,
        )

        ret_control_idx = torch.argmax(opt_vals / costs).item()
        ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query


class UCB_PSQ(Acquisition):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        def ucb(X):
            posterior = gp.posterior(X=X)
            mean = posterior.mean
            variance = posterior.variance
            return mean + self.beta * torch.sqrt(variance)

        dims = bounds.shape[-1]
        skip_expectations = False

        for i, control_set in enumerate(control_sets):
            if len(control_set) == dims:
                skip_expectations = True
                ret_control_idx = i
                ret_query, _ = maximize_fn(f=ucb, n_warmup=10000, bounds=bounds)
                ret_query = ret_query[None, :]
                break

        if not skip_expectations:
            opt_queries, opt_vals = get_opt_queries_and_vals(
                f=ucb,
                control_sets=control_sets,
                random_sets=random_sets,
                all_dists_samples=all_dists_samples,
                bounds=bounds,
            )
            ret_control_idx = torch.argmax(opt_vals).item()
            ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query


class UCB_PSQ_CS(Acquisition):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        def ucb(X):
            posterior = gp.posterior(X=X)
            mean = posterior.mean
            variance = posterior.variance
            return mean + self.beta * torch.sqrt(variance)

        # dims = bounds.shape[-1]
        # skip_expectations = False
        # for i, control_set in enumerate(control_sets):
        #     if len(control_set) == dims and np.allclose(eps, 0):
        #         skip_expectations = True
        #         ret_control_idx = i
        #         ret_query, _ = maximize_fn(f=ucb, n_warmup=10000, bounds=bounds)
        #         ret_query = ret_query[None, :]
        #         break
        #
        # if not skip_expectations:
        #     opt_queries, opt_vals = get_opt_queries_and_vals(
        #         f=ucb,
        #         control_sets=control_sets,
        #         random_sets=random_sets,
        #         all_dists_samples=all_dists_samples,
        #         bounds=bounds,
        #     )
        #     max_val = torch.max(opt_vals)
        #     ret_control_idx = torch.argmax(opt_vals).item()
        #     for i in range(len(opt_vals)):
        #         if opt_vals[i] + eps >= max_val:
        #             ret_control_idx = i
        #             break
        #     ret_query = opt_queries[ret_control_idx]
        #
        # return ret_control_idx, ret_query

        opt_queries, opt_vals = get_opt_queries_and_vals(
            f=ucb,
            control_sets=control_sets,
            random_sets=random_sets,
            all_dists_samples=all_dists_samples,
            bounds=bounds,
        )

        eps = eps_schedule.next(opt_vals=opt_vals)

        max_val = torch.max(opt_vals)
        ret_control_idx = torch.argmax(opt_vals).item()
        for i in range(len(opt_vals)):
            if opt_vals[i] + eps >= max_val:
                ret_control_idx = i
                break
        ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query


class UCB_PSQ_Naive(Acquisition):
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def acquire(
        self,
        train_X,
        train_y,
        gp,
        control_sets,
        random_sets,
        all_dists_samples,
        bounds,
        eps_schedule,
        costs,
    ):
        def ucb(X):
            posterior = gp.posterior(X=X)
            mean = posterior.mean
            variance = posterior.variance
            return mean + self.beta * torch.sqrt(variance)

        opt_queries, opt_vals = get_opt_queries_and_vals(
            f=ucb,
            control_sets=control_sets,
            random_sets=random_sets,
            all_dists_samples=all_dists_samples,
            bounds=bounds,
        )
        ret_control_idx = torch.argmax(opt_vals / costs).item()
        ret_query = opt_queries[ret_control_idx]

        return ret_control_idx, ret_query

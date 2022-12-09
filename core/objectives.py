from botorch import test_functions
from marchantia.core.synth_func import create_synth_funcs_combined
import torch

from core.gp import sample_gp_prior
from core.utils import maximize_fn, log


def get_objective(objective_name, noise_std, is_input_transform, kernel, dims):
    """
    Get objective function, bounds and its max function value (for regret).
    :param config_name: str.
    :param objective_name: str.
    :param noise_std: float.
    :param is_input_transform: bool. Set to True to transform the domain to the unit hypercube.
    :param dtype: Torch dtype.
    :return: objective function Callable that takes in arrays of shape (..., d) and returns an array of shape (..., 1),
    bounds with shape (2, d), optimal function value.
    """
    if objective_name == "gpsample":
        bounds = torch.stack([torch.zeros(dims), torch.ones(dims)])

        obj_func = sample_gp_prior(kernel=kernel, bounds=bounds, num_points=1000)

        _, opt_val = maximize_fn(f=obj_func, bounds=bounds, n_warmup=10000,)

    elif objective_name == "hartmann":
        neg_obj = test_functions.Hartmann(dim=6, negate=True)
        bounds = neg_obj.bounds.to(dtype=torch.double)
        unsqueezed_obj = lambda x: neg_obj(x).unsqueeze(-1)
        if is_input_transform:
            obj_func = input_transform_wrapper(obj_func=unsqueezed_obj, bounds=bounds)
            bounds = torch.stack(
                [torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]
            )
        else:
            obj_func = unsqueezed_obj

        opt_val = neg_obj.optimal_value

    elif objective_name == "plant":
        bounds = torch.tensor(
            [[0, 7.7], [0, 3.5], [0, 10.4], [8.9, 11.3], [2.5, 6.5]], dtype=torch.double
        ).T
        leafarea_meanvar_func = create_synth_funcs_combined(standardize=True)[0]
        obj_func = lambda x: torch.tensor(
            leafarea_meanvar_func(x.numpy())[0], dtype=torch.double
        )
        if is_input_transform:
            obj_func = input_transform_wrapper(obj_func=obj_func, bounds=bounds)
            bounds = torch.stack(
                [torch.zeros_like(bounds[0]), torch.ones_like(bounds[1])]
            )

        _, opt_val = maximize_fn(f=obj_func, n_warmup=10000, bounds=bounds)
    else:
        raise Exception("Incorrect obj_name passed to get_objective")

    noisy_obj_func = noisy_wrapper(obj_func=obj_func, noise_std=noise_std)
    return obj_func, noisy_obj_func, opt_val, bounds


def input_transform_wrapper(obj_func, bounds):
    """
    Wrapper around an existing objective function. Changes the bounds of the objective function to be the d-dim
    unit hypercube [0, 1]^d.
    :param obj_func: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    :param bounds: array of shape (2, d).
    :return: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    """
    return lambda x: obj_func(input_transform(x, bounds))


def input_transform(x, bounds):
    return x * (bounds[1] - bounds[0]) + bounds[0]


def noisy_wrapper(obj_func, noise_std):
    """
    Wrapper around an existing objective function. Turns a noiseless objective function into a noisy one.
    :param obj_func: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    :param noise_std: float.
    :return: Callable that takes in an array of shape (..., d) and returns an array of shape (..., 1).
    """
    return lambda x: obj_func(x) + noise_std * torch.randn(size=x.shape[:-1] + (1,))

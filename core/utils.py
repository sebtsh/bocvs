import datetime
import glob
import numpy as np
import pickle
from scipy.optimize import minimize
import torch


def log(msg):
    print(str(datetime.datetime.now()) + " - " + msg)


def uniform_samples(bounds, num_samples, dtype):
    low = bounds[0]
    high = bounds[1]
    d = len(low)
    return torch.rand(size=(num_samples, d), dtype=dtype) * (high - low) + low


def get_missing_idxs(num_idxs, given_idxs):
    all_idxs = torch.arange(num_idxs)
    combined = torch.cat((all_idxs, given_idxs))
    uniques, counts = combined.unique(return_counts=True)
    return torch.sort(uniques[counts == 1])[0]


def minmax_normalize(arr):
    b = torch.max(arr)
    a = torch.min(arr)
    return (arr - a) / (b - a)


def load_most_recent_state(inter_save_dir, filename):
    path = inter_save_dir + f"{filename}-iter*.p"
    save_list = glob.glob(path)
    if len(save_list) == 0:
        print(f"No save states with path {path} found")
        train_X, train_y, state_dict, max_iter = None, None, None, None
    else:
        # Get most recent
        max_iter = 0
        for i, save_path in enumerate(save_list):
            cur_iter = int(
                save_path[
                    len(inter_save_dir)
                    + len(filename)
                    + len("-iter") : len(save_path)
                    - len(".p")
                ]
            )
            if cur_iter > max_iter:
                most_recent_idx = i
                max_iter = cur_iter
        most_recent_path = save_list[most_recent_idx]
        print(f"Loading iter {max_iter} from {most_recent_path}")
        train_X, train_y, state_dict = pickle.load(open(most_recent_path, "rb"))

    return train_X, train_y, state_dict, max_iter


def maximize_fn(
    f, bounds, dtype, num_warmup=10000, num_iter=10
):
    """
    Approximately maximizes a function f using sampling + L-BFGS-B method adapted from
    https://github.com/fmfn/BayesianOptimization.
    :param f: Callable that takes in an array of shape (n, d) and returns an array of shape (n, 1).
    :param bounds: Array of shape (2, d). Lower and upper bounds of each variable.
    :param dtype:
    :param num_warmup: int. Number of random samples.
    :param num_iter: int. Number of L-BFGS-B starting points.
    :return: (Array of shape (d,), max_val).
    """
    neg_func_squeezed = lambda x: np.squeeze((-f(torch.tensor(x[None, :]))).cpu().detach().numpy())

    # Random sampling
    x_tries = uniform_samples(bounds=bounds,
                              num_samples=num_warmup,
                              dtype=dtype)
    f_x = torch.squeeze(f(x_tries), dim=1)
    x_max = x_tries[np.argmax(f_x)]
    f_max = np.max(f_x)

    # L-BFGS-B
    x_seeds = uniform_samples(bounds=bounds,
                              num_samples=num_iter-1,
                              dtype=dtype)
    x_seeds = torch.cat([x_seeds, x_max[None, :]], dim=0)
    x_seeds_np = x_seeds.cpu().detach().numpy()
    for x_try in x_seeds_np:
        res = minimize(
            fun=neg_func_squeezed,
            x0=x_try,
            bounds=bounds,
            method="L-BFGS-B",
        )
        if not res.success:
            continue
        if -res.fun >= f_max:
            x_max = res.x
            f_max = -res.fun
    f_argmax = np.clip(x_max, bounds[:, 0], bounds[:, 1])
    return torch.tensor(f_argmax), f_max

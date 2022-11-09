from botorch.optim import optimize_acqf
import pickle
import torch
from tqdm import trange

from core.acquisitions import get_acquisition


def bo_loop(
    train_X,
    train_y,
    gp,
    obj_func,
    start_iter,
    num_iters,
    acq_name,
    bounds,
    filename,
    inter_save_dir,
):
    for t in trange(start_iter, num_iters):
        acquisition = get_acquisition(acq_name=acq_name, gp=gp, train_y=train_y)

        x_t, _ = optimize_acqf(
            acq_function=acquisition,
            bounds=bounds,
            q=1,
            num_restarts=5,
            raw_samples=20,
        )  # (1, d)

        y_t = obj_func(x_t)  # (1 ,1)
        train_X = torch.cat([train_X, x_t])
        train_y = torch.cat([train_y, y_t])

        # Save state every 50 iterations
        if t != 0 and t % 50 == 0:
            pickle.dump(
                (train_X, train_y),
                open(inter_save_dir + f"{filename}-iter{t}.p", "wb"),
            )

    return train_X, train_y

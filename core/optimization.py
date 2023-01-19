import numpy as np
import torch

from core.dists import sample_from_random_sets
from core.gp import ExactGPModel
from core.utils import log


def bo_loop(
    train_X,
    train_y,
    likelihood,
    kernel,
    noisy_obj_func,
    start_iter,
    budget,
    acquisition,
    bounds,
    all_dists,
    control_sets,
    random_sets,
    all_dists_samples,
    costs,
    eps_schedule,
):
    control_set_idxs = []
    control_queries = []
    all_eps = []
    remaining_budget = budget
    t = start_iter

    while True:
        log(f"t = {t}, remaining budget: {remaining_budget}")

        gp = ExactGPModel(train_X, torch.squeeze(train_y), kernel, likelihood)
        gp.eval()
        control_set_idx, control_query = acquisition.acquire(
            train_X=train_X,
            train_y=train_y,
            gp=gp,
            all_dists_samples=all_dists_samples,
            control_sets=control_sets,
            random_sets=random_sets,
            bounds=bounds,
            eps_schedule=eps_schedule,
            costs=costs,
        )
        log(f"Control set chosen: {control_set_idx}")

        # Exit condition
        cost = costs[control_set_idx]
        if cost > remaining_budget:
            break

        # Observation
        dims = bounds.shape[-1]
        if len(control_sets[control_set_idx]) == dims:
            x_t = control_query
        else:
            control_set = control_sets[control_set_idx]
            random_set = random_sets[control_set_idx]
            cat_idxs = np.concatenate([control_set, random_set])
            order_idxs = np.array(
                [np.where(cat_idxs == j)[0][0] for j in np.arange(len(cat_idxs))]
            )
            random_query = sample_from_random_sets(
                all_dists=all_dists, random_set=random_set
            )
            unordered_x_t = torch.cat([control_query, random_query], dim=-1)
            x_t = unordered_x_t[:, order_idxs]
        y_t = noisy_obj_func(x_t)  # (1 ,1)

        # Update datasets
        train_X = torch.cat([train_X, x_t])
        train_y = torch.cat([train_y, y_t])
        control_set_idxs.append(control_set_idx)
        control_queries.append(control_query)

        # Epsilon schedule management
        if eps_schedule is not None:
            eps_schedule.update()
            all_eps.append(eps_schedule.last_eps)

        # Loop management
        t += 1
        remaining_budget = remaining_budget - cost

    return train_X, train_y, control_set_idxs, control_queries, t, all_eps

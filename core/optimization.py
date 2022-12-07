from botorch.models import SingleTaskGP
import pickle
import torch

from core.acquisitions import get_acquisition
from core.dists import sample_from_random_sets
from core.utils import log


def bo_loop(
    train_X,
    train_y,
    likelihood,
    kernel,
    noisy_obj_func,
    start_iter,
    budget,
    acq_name,
    bounds,
    all_dists,
    control_sets,
    random_sets,
    all_dists_samples,
    costs,
    eps_schedule,
    filename,
    inter_save_dir,
):
    control_set_idxs = []
    control_queries = []
    all_eps = []
    remaining_budget = budget
    t = start_iter
    while True:
        log(f"t = {t}, remaining budget: {remaining_budget}")
        # Acquire next query
        gp = SingleTaskGP(
            train_X=train_X, train_Y=train_y, likelihood=likelihood, covar_module=kernel
        )

        acquisition = get_acquisition(acq_name=acq_name)

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

        # Exit condition
        cost = costs[control_set_idx]
        if cost > remaining_budget:
            break

        # Observation
        dims = bounds.shape[-1]
        if len(control_sets[control_set_idx]) == dims:
            x_t = control_query
        else:
            random_query = sample_from_random_sets(
                all_dists=all_dists, random_set=random_sets[control_set_idx]
            )
            x_t = torch.cat([control_query, random_query], dim=-1)
        y_t = noisy_obj_func(x_t)  # (1 ,1)

        # Update datasets
        train_X = torch.cat([train_X, x_t])
        train_y = torch.cat([train_y, y_t])
        control_set_idxs.append(control_set_idx)
        control_queries.append(control_query)

        # Epsilon schedule management
        eps_schedule.update(prev_control_idx=control_set_idx)
        all_eps.append(eps_schedule.last_eps)

        # Loop management
        t += 1
        remaining_budget = remaining_budget - cost

        # Save state every 50 iterations
        # if t != 0 and t % 50 == 0:
        #     pickle.dump(
        #         (
        #             train_X,
        #             train_y,
        #             control_set_idxs,
        #             control_queries,
        #             t,
        #             remaining_budget,
        #         ),
        #         open(inter_save_dir + f"{filename}-iter{t}.p", "wb"),
        #     )

    return train_X, train_y, control_set_idxs, control_queries, t, all_eps

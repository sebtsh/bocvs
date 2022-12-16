from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, RBFKernel
import matplotlib
import numpy as np
from pathlib import Path
import pickle
import sys
import torch

from core.dists import get_dists_and_samples, get_marginal_var
from core.objectives import get_objective
from core.optimization import bo_loop
from core.psq import get_control_sets_and_costs, get_eps_schedule
from core.regret import get_regret, plot_regret
from core.utils import log, uniform_samples, load_most_recent_state


matplotlib.use("Agg")

job_id = sys.argv[1]
job_dir = "jobs/"
job_filename = job_dir + f"job{job_id}.txt"


def parse_params(job_file):
    """
    WARNING: assumes all ids and seeds have only 1 digit!
    :param job_file:
    :return:
    """
    missing_filenames = job_file.readlines()
    all_params = []
    for filename in missing_filenames:
        params = filename.split(sep="_")
        obj_name = params[0]
        acq_name = params[1]
        eps_schedule_id = int(params[2][-1])
        costs_id = int(params[3][-1])
        var_id = int(params[4][-1])
        budget = int(params[5][1:])
        seed = int(params[6][4])
        all_params.append(
            (obj_name, acq_name, eps_schedule_id, costs_id, var_id, budget, seed)
        )

    return all_params


def main(
    obj_name,
    acq_name,
    eps_schedule_id,
    costs_id,
    var_id,
    budget,
    seed,
    dims,
    noise_std,
    init_lengthscale,
    n_init_points,
    load_state,
):
    args = dict(locals().items())
    log(f"Running with parameters {args}")

    log(f"======== NEW RUN ========")
    log(
        (
            f"{obj_name}, {acq_name}, eps_sched:{eps_schedule_id}, cost_id:{costs_id}"
            f", var_id:{var_id}, C:{budget}, seed:{seed}"
        )
    )
    torch.manual_seed(seed)

    # Directory for saving results
    base_dir = "results/" + obj_name + "/"
    pickles_save_dir = base_dir + "pickles/"
    figures_save_dir = base_dir + "figures/"
    inter_save_dir = base_dir + "inter/"
    Path(pickles_save_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_save_dir).mkdir(parents=True, exist_ok=True)
    Path(inter_save_dir).mkdir(parents=True, exist_ok=True)
    filename = (
        f"{obj_name}_{acq_name}_es{eps_schedule_id}_c{costs_id}"
        f"_var{var_id}_C{budget}_seed{seed}"
    )
    filename = filename.replace(".", ",")

    # Objective function
    if obj_name == "gpsample":  # If sampling from GP, we need to define kernel first
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dims))
        kernel.outputscale = 1.0
        kernel.base_kernel.lengthscale = init_lengthscale
    else:
        kernel = None

    obj_func, noisy_obj_func, opt_val_det, bounds = get_objective(
        objective_name=obj_name,
        noise_std=noise_std,
        is_input_transform=True,
        kernel=kernel,
        dims=dims,
    )
    log(f"opt_val: {opt_val_det}")

    # Initialize state
    if load_state:
        raise NotImplementedError
        # init_X, init_y, state_dict, max_iter = load_most_recent_state(
        #     inter_save_dir=inter_save_dir, filename=filename
        # )
        # start_iter = max_iter + 1
        # if max_iter is None:  # if max_iter is None, no save states found
        #     load_state = False
    else:
        log("Starting new run from iter 0")
        start_iter = 0
        state_dict = None
        # Initial data
        init_X = uniform_samples(bounds=bounds, n_samples=n_init_points)
        init_y = noisy_obj_func(init_X)

    # GP parameters
    if obj_name != "gpsample":
        dims = bounds.shape[-1]
        kernel = ScaleKernel(RBFKernel(ard_num_dims=dims))
        kernel.outputscale = 1.0
        kernel.base_kernel.lengthscale = init_lengthscale

    likelihood = GaussianLikelihood()
    likelihood.noise = noise_std**2

    # Control/random sets and costs
    control_sets, random_sets, costs = get_control_sets_and_costs(
        dims=dims, control_sets_id=0, costs_id=costs_id
    )
    marginal_var = get_marginal_var(var_id=var_id)
    all_dists, all_dists_samples = get_dists_and_samples(
        dims=dims, variance=marginal_var
    )
    variances = marginal_var * np.ones(dims, dtype=np.double)
    lengthscales = init_lengthscale * np.ones(dims, dtype=np.double)
    eps_schedule = get_eps_schedule(
        id=eps_schedule_id,
        costs=costs,
        control_sets=control_sets,
        random_sets=random_sets,
        variances=variances,
        lengthscales=lengthscales,
        budget=budget,
    )

    # Optimization loop
    (final_X, final_y, control_set_idxs, control_queries, T, all_eps,) = bo_loop(
        train_X=init_X,
        train_y=init_y,
        likelihood=likelihood,
        kernel=kernel,
        noisy_obj_func=noisy_obj_func,
        start_iter=start_iter,
        budget=budget,
        acq_name=acq_name,
        bounds=bounds,
        all_dists=all_dists,
        control_sets=control_sets,
        random_sets=random_sets,
        all_dists_samples=all_dists_samples,
        costs=costs,
        eps_schedule=eps_schedule,
        filename=filename,
        inter_save_dir=inter_save_dir,
    )
    # Regret
    log("Calculating regret")
    simple_regret, cumu_regret, cs_cumu_regret, cost_per_iter = get_regret(
        control_set_idxs=control_set_idxs,
        control_queries=control_queries,
        obj_func=obj_func,
        control_sets=control_sets,
        random_sets=random_sets,
        all_dists_samples=all_dists_samples,
        bounds=bounds,
        costs=costs,
    )

    plot_regret(
        regret=cumu_regret,
        cost_per_iter=cost_per_iter,
        x_axis="T",
        num_iters=T,
        save=True,
        save_dir=figures_save_dir,
        filename=filename + "_T",
    )

    plot_regret(
        regret=cs_cumu_regret,
        cost_per_iter=cost_per_iter,
        x_axis="C",
        num_iters=T,
        save=True,
        save_dir=figures_save_dir,
        filename=filename + "_C",
    )

    plot_regret(
        regret=simple_regret,
        cost_per_iter=cost_per_iter,
        x_axis="C",
        num_iters=T,
        save=True,
        save_dir=figures_save_dir,
        filename=filename + "_Csimple",
    )

    # Save results
    pickle.dump(
        (
            final_X,
            final_y,
            control_set_idxs,
            control_queries,
            all_dists_samples,
            simple_regret,
            cumu_regret,
            cs_cumu_regret,
            cost_per_iter,
            T,
            args,
        ),
        open(pickles_save_dir + f"{filename}.p", "wb"),
    )

    print("cumu_regret:")
    print(cumu_regret)
    print("final y:")
    print(torch.squeeze(final_y))
    print("all_eps:")
    print(all_eps)
    print("control_set_idxs:")
    print(control_set_idxs)
    print("count of each control set played:")
    print(np.histogram(control_set_idxs, bins=np.arange(len(control_sets) + 1))[0])

    log(f"Completed run with parameters {args}")


with open(job_filename) as f:
    all_params = parse_params(f)
    print(f"all params: {all_params}")

for params in all_params:  # run each experiment sequentially
    obj_name, acq_name, eps_schedule_id, costs_id, var_id, budget, seed = params

    print("Getting configs")

    if obj_name == "gpsample":
        dims = 3
        noise_std = 0.01
        init_lengthscale = 0.1
        n_init_points = 5
        load_state = False
    elif obj_name == "hartmann":
        dims = 6
        noise_std = 0.01
        init_lengthscale = 0.2
        n_init_points = 5
        load_state = False
    elif obj_name == "plant":
        dims = 5
        noise_std = 0.01
        init_lengthscale = 0.2
        n_init_points = 5
        load_state = False
    else:
        raise NotImplementedError

    main(
        obj_name=obj_name,
        acq_name=acq_name,
        eps_schedule_id=eps_schedule_id,
        costs_id=costs_id,
        var_id=var_id,
        budget=budget,
        seed=seed,
        dims=dims,
        noise_std=noise_std,
        init_lengthscale=init_lengthscale,
        n_init_points=n_init_points,
        load_state=load_state,
    )

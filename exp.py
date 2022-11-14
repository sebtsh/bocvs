from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.kernels import ScaleKernel, MaternKernel
import matplotlib
from pathlib import Path
import pickle
from sacred import Experiment
from sacred.observers import FileStorageObserver
import torch

from core.dists import get_dists_and_samples, get_opt_queries_and_vals
from core.objectives import get_objective
from core.optimization import bo_loop
from core.psq import get_control_sets_and_costs
from core.regret import get_simple_regret, plot_regret
from core.utils import log, uniform_samples, load_most_recent_state


matplotlib.use("Agg")
ex = Experiment("CS-PSQ-BO")
ex.observers.append(FileStorageObserver("./runs"))


@ex.named_config
def gpsample():
    config_name = "gpsample"
    obj_name = "gpsample"
    acq_name = "ei"
    dims = 6
    control_sets_id = 0
    costs_id = 0
    eps_schedule_id = 0
    marginal_var = 0.04
    noise_std = 0.01
    init_lengthscale = 0.2
    num_init_points = 5
    num_iters = 200
    budget = 1000
    seed = 0
    load_state = False


@ex.named_config
def synth():
    config_name = "synth"
    obj_name = "dixonprice"
    acq_name = "ei"
    dims = None
    control_sets_id = 0
    costs_id = 0
    eps_schedule_id = 0
    marginal_var = 0.04
    noise_std = 0.01
    init_lengthscale = 0.2
    num_init_points = 5
    num_iters = 200
    budget = 1000
    seed = 0
    load_state = False


@ex.automain
def main(
    config_name,
    obj_name,
    acq_name,
    dims,
    control_sets_id,
    costs_id,
    eps_schedule_id,
    marginal_var,
    noise_std,
    init_lengthscale,
    num_init_points,
    num_iters,
    budget,
    seed,
    load_state,
):
    args = dict(locals().items())
    log(f"Running with parameters {args}")
    run_id = ex.current_run._id
    torch.manual_seed(seed)

    # Directory for saving results
    base_dir = "results/" + config_name + "/"
    pickles_save_dir = base_dir + "pickles/"
    figures_save_dir = base_dir + "figures/"
    inter_save_dir = base_dir + "inter/"
    Path(pickles_save_dir).mkdir(parents=True, exist_ok=True)
    Path(figures_save_dir).mkdir(parents=True, exist_ok=True)
    Path(inter_save_dir).mkdir(parents=True, exist_ok=True)
    filename = f"{config_name}_{obj_name}_{acq_name}_seed-{seed}"

    # Objective function
    if config_name is "gpsample":  # If sampling from GP, we need to define kernel first
        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dims))
        kernel.base_kernel.lengthscale = init_lengthscale
    else:
        kernel = None

    obj_func, noisy_obj_func, opt_val_det, bounds = get_objective(
        config_name=config_name,
        objective_name=obj_name,
        noise_std=noise_std,
        is_input_transform=True,
        kernel=kernel,
        dims=dims,
    )

    # Initialize state
    if load_state:
        init_X, init_y, state_dict, max_iter = load_most_recent_state(
            inter_save_dir=inter_save_dir, filename=filename
        )
        start_iter = max_iter + 1
        if max_iter is None:  # if max_iter is None, no save states found
            load_state = False
    else:
        log("Starting new run from iter 0")
        start_iter = 0
        state_dict = None
        # Initial data
        init_X = uniform_samples(bounds=bounds, n_samples=num_init_points)
        init_y = obj_func(init_X)

    # GP parameters
    if config_name is not "gpsample":
        dims = bounds.shape[-1]
        kernel = ScaleKernel(MaternKernel(nu=2.5, ard_num_dims=dims))
        kernel.base_kernel.lengthscale = init_lengthscale

    likelihood = GaussianLikelihood()
    likelihood.noise = noise_std

    # Control/random sets and costs
    control_sets, random_sets, costs = get_control_sets_and_costs(
        dims=dims, control_sets_id=control_sets_id, costs_id=costs_id
    )
    all_dists, all_dists_samples = get_dists_and_samples(
        dims=dims, variance=marginal_var
    )
    _, opt_vals = get_opt_queries_and_vals(
        f=obj_func,
        control_sets=control_sets,
        random_sets=random_sets,
        all_dists_samples=all_dists_samples,
        bounds=bounds,
    )

    # Optimization loop
    final_X, final_y = bo_loop(
        train_X=init_X,
        train_y=init_y,
        likelihood=likelihood,
        kernel=kernel,
        obj_func=noisy_obj_func,
        start_iter=start_iter,
        num_iters=num_iters,
        acq_name=acq_name,
        bounds=bounds,
        filename=filename,
        inter_save_dir=inter_save_dir,
    )
    # Regret
    chosen_X = final_X[num_init_points:]
    simple_regret = get_simple_regret(X=chosen_X, obj_func=obj_func, opt_val=opt_val)
    plot_regret(
        regret=simple_regret,
        num_iters=num_iters,
        save=True,
        save_dir=figures_save_dir,
        filename=filename,
    )

    # Save results
    pickle.dump(
        (final_X, final_y, chosen_X, simple_regret, args),
        open(pickles_save_dir + f"{filename}.p", "wb"),
    )

    log(f"Completed run {run_id} with parameters {args}")

import pickle
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sacred import Experiment
from sacred.observers import FileStorageObserver

from core.regret import (
    interpolate_regret,
    first_index_of_conv,
    get_sample_regret_from_cumu,
)
from core.utils import find_nearest

ex = Experiment("CSPBO_results")
ex.observers.append(FileStorageObserver("../runs"))


@ex.named_config
def gpsample():
    obj_name = "gpsample"
    budget = 50
    num_seeds = 10
    legend_loc = "best"


@ex.named_config
def hartmann():
    obj_name = "hartmann"
    budget = 50
    num_seeds = 10
    legend_loc = "best"


@ex.named_config
def plant():
    obj_name = "plant"
    budget = 200
    num_seeds = 10
    legend_loc = "best"


@ex.named_config
def airfoil():
    obj_name = "airfoil"
    budget = 50
    num_seeds = 10
    legend_loc = "best"


@ex.automain
def main(
    obj_name,
    budget,
    num_seeds,
    legend_loc,
    figsize=(20, 20),
    dpi=200,
):
    text_size = 16
    tick_size = 10
    base_dir = "results/" + obj_name + "/"
    save_dir = "summary_results/"
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    pickles_dir = base_dir + "pickles/"

    acquisitions = ["ts", "ucb", "etc_es0", "etc_es1", "etc_es2"]

    color_dict = {
        "ts": "black",
        "ucb": "#d7263d",
        "etc_es0": "#fbb13c",
        "etc_es1": "#26c485",
        "etc_es2": "#00a6ed",
        "ts-naive_es0": "grey",
        "ucb-naive_es0": "brown",
        "ei_es0": "purple",
    }
    acq_name_dict = {
        "ts": "TS",
        "ucb": "UCB-PSQ",
        "etc_es0": "ETC-50",
        "etc_es1": "ETC-100",
        "etc_es2": "ETC-Ada",
        "ts-naive_es0": "TS-PSQ per cost",
        "ucb-naive_es0": "UCB-PSQ per cost",
        "ei_es0": "EI-PSQ per cost",
    }
    print(f"================ {obj_name} ================")
    seeds = list(range(num_seeds))

    fig_simple, all_axs_simple = plt.subplots(
        3, 3, figsize=figsize, dpi=dpi, sharey="row", sharex="col"
    )
    fig_cumu, all_axs_cumu = plt.subplots(3, 5, figsize=figsize, dpi=dpi)

    final_regrets_dict = {}

    for costs_id in range(3):
        print(f"======== costs_id: {costs_id} ========")
        costs_dict = {0: "Cheap", 1: "Moderate", 2: "Expensive"}
        costs_alias = costs_dict[costs_id]
        for var_id in range(3):
            print(f"==== var_id: {var_id} ====")
            var_dict = {0: 0.02, 1: 0.04, 2: 0.08}
            variance = var_dict[var_id]
            axs_simple = all_axs_simple[costs_id][var_id - 2]
            axs_cumu = all_axs_cumu[costs_id][var_id - 2]

            axs_simple.grid(which="major")
            axs_simple.grid(which="minor", linestyle=":", alpha=0.3)
            axs_cumu.grid(which="major")
            axs_cumu.grid(which="minor", linestyle=":", alpha=0.3)

            for acquisition in acquisitions:
                print(f"== {acquisition} ==")
                if acquisition == "ucb" or acquisition == "ts":
                    acq_alias = acquisition + "_es0"
                    if obj_name == "airfoil":
                        virtual_costs_id = costs_id
                        virtual_var_id = var_id
                    else:
                        virtual_costs_id = 0
                        virtual_var_id = 0
                else:
                    acq_alias = acquisition
                    virtual_costs_id = costs_id
                    virtual_var_id = var_id
                virtual_budget = budget

                color = color_dict[acquisition]
                all_cost_per_iter_cumusums = []
                all_simple_regrets = []
                all_cumu_regrets = []
                all_first_budgets = []
                all_first_iters = []
                all_T = []
                all_mean_sample_regret_with_best = []

                for i, seed in enumerate(seeds):
                    filename = (
                        f"{obj_name}_{acq_alias}_c{virtual_costs_id}"
                        f"_var{virtual_var_id}_C{virtual_budget}_seed{seed}"
                    )
                    filename = filename.replace(".", ",") + ".p"
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
                    ) = pickle.load(open(pickles_dir + filename, "rb"))
                    first_index = first_index_of_conv(control_set_idxs, 6)
                    first_budget = np.cumsum(cost_per_iter)[first_index]
                    best_idxs = np.where(np.array(control_set_idxs) == 6)[0]
                    sample_regret = get_sample_regret_from_cumu(cumu_regret)

                    all_cost_per_iter_cumusums.append(np.cumsum(cost_per_iter))
                    all_simple_regrets.append(simple_regret)
                    all_cumu_regrets.append(cs_cumu_regret)
                    all_first_iters.append(first_index + 1)
                    all_first_budgets.append(first_budget)
                    all_T.append(len(control_queries))
                    if len(best_idxs) != 0:
                        all_mean_sample_regret_with_best.append(
                            np.sum(sample_regret[best_idxs])
                        )

                interpolated_all_simple_regrets, _ = interpolate_regret(
                    regrets=all_simple_regrets,
                    all_cost_per_iter_cumusums=all_cost_per_iter_cumusums,
                )
                interpolated_all_cumu_regrets, cost_axis = interpolate_regret(
                    regrets=all_cumu_regrets,
                    all_cost_per_iter_cumusums=all_cost_per_iter_cumusums,
                )

                mean_simple_regrets = np.mean(interpolated_all_simple_regrets, axis=0)
                std_err_simple_regrets = np.std(
                    interpolated_all_simple_regrets, axis=0
                ) / np.sqrt(num_seeds)

                mean_cumu_regrets = np.mean(interpolated_all_cumu_regrets, axis=0)
                std_err_cumu_regrets = np.std(
                    interpolated_all_cumu_regrets, axis=0
                ) / np.sqrt(num_seeds)

                # average time after which the algorithm ONLY chooses the
                # best control set
                mean_first_budget = np.mean(all_first_budgets)

                # cut cost at budget
                found_limit = False
                for j, c in enumerate(cost_axis):
                    if c > budget:
                        limit = j  # first index at which budget is exceeded
                        found_limit = True
                        break
                if found_limit:
                    cost_axis = cost_axis[:limit]
                    mean_simple_regrets = mean_simple_regrets[:limit]
                    std_err_simple_regrets = std_err_simple_regrets[:limit]
                    mean_cumu_regrets = mean_cumu_regrets[:limit]
                    std_err_cumu_regrets = std_err_cumu_regrets[:limit]

                final_regret = mean_simple_regrets[-1]
                key = (
                    f"{obj_name}_{acq_alias}_c{virtual_costs_id}"
                    f"_var{virtual_var_id}_C{virtual_budget}"
                )
                final_regrets_dict[key] = final_regret

                acq_name = acq_name_dict[acquisition]
                mean_first_budget_idx = find_nearest(cost_axis, mean_first_budget)
                axs_simple.plot(
                    cost_axis,
                    mean_simple_regrets,
                    markevery=[mean_first_budget_idx],
                    marker="D",
                    label=acq_name,
                    color=color,
                )
                axs_simple.fill_between(
                    cost_axis,
                    mean_simple_regrets - std_err_simple_regrets,
                    mean_simple_regrets + std_err_simple_regrets,
                    alpha=0.2,
                    color=color,
                )

                axs_cumu.plot(cost_axis, mean_cumu_regrets, label=acq_name, color=color)
                axs_cumu.fill_between(
                    cost_axis,
                    mean_cumu_regrets - std_err_cumu_regrets,
                    mean_cumu_regrets + std_err_cumu_regrets,
                    alpha=0.2,
                    color=color,
                )

                axs_simple.set_title(
                    f"{costs_alias} costs, variance={variance}", size=text_size
                )
                if costs_id == 1:
                    axs_simple.set_xlabel("Budget $C$", size=text_size)
                if var_id == 2:
                    axs_simple.set_ylabel("Simple regret", size=text_size)
                if obj_name == "plant":
                    axs_simple.set_xticks([0, 40, 80, 120, 160, 200])
                axs_simple.tick_params(labelsize=tick_size)
                axs_simple.legend(fontsize=text_size - 2, loc=legend_loc)
                axs_simple.set_yscale("log")

                axs_cumu.set_title(
                    f"{costs_alias} costs, variance={variance}", size=text_size
                )
                axs_cumu.set_xlabel("Budget $C$", size=text_size)
                axs_cumu.set_ylabel("Cumulative regret", size=text_size)
                axs_cumu.tick_params(labelsize=tick_size)
                axs_cumu.legend(fontsize=text_size - 2, loc=legend_loc)

    fig_simple.tight_layout()
    fig_simple.savefig(
        save_dir + f"{obj_name}-simple_regret.png",
        dpi=dpi,
        bbox_inches="tight",
        format="png",
    )

    fig_cumu.tight_layout()
    fig_cumu.savefig(
        save_dir + f"{obj_name}-cumu_regret.pdf",
        dpi=dpi,
        bbox_inches="tight",
        format="pdf",
    )

    pickle.dump(final_regrets_dict, open(f"{obj_name}_finalregs.p", "wb"))

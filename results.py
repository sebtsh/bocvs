import pickle
import matplotlib.pyplot as plt
import numpy as np
from sacred import Experiment
from sacred.observers import FileStorageObserver

from core.regret import interpolate_regret

ex = Experiment("CSPBO_results")
ex.observers.append(FileStorageObserver("../runs"))


@ex.named_config
def gpsample():
    obj_name = "gpsample"
    budget = 100
    num_seeds = 5
    legend_loc = "best"


@ex.named_config
def hartmann():
    obj_name = "hartmann"
    budget = 500
    num_seeds = 5
    legend_loc = "best"


@ex.named_config
def plant():
    obj_name = "plant"
    budget = 500
    num_seeds = 5
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
    save_dir = base_dir
    pickles_dir = base_dir + "pickles/"

    acquisitions = ["ucb-cs_es0", "ucb-cs_es1", "ucb", "ts"]
    color_dict = {
        "ts": "#d7263d",
        "ucb": "#fbb13c",
        "ucb-cs_es0": "#00a6ed",
        "ucb-cs_es1": "#26c485",
    }
    acq_name_dict = {
        "ts": "TS",
        "ucb": "UCB-PSQ",
        "ucb-cs_es0": "UCB-PSQ-CS (Ada)",
        "ucb-cs_es1": "UCB-PSQ-CS (Lin1)",
    }

    seeds = list(range(num_seeds))

    fig_simple, all_axs_simple = plt.subplots(3, 3, figsize=figsize, dpi=dpi)
    fig_cumu, all_axs_cumu = plt.subplots(3, 3, figsize=figsize, dpi=dpi)

    for costs_id in range(3):
        costs_dict = {0: "Cheap", 1: "Moderate", 2: "Expensive"}
        costs_alias = costs_dict[costs_id]
        for var_id in range(3):
            var_dict = {0: 0.01, 1: 0.04, 2: 0.08}
            variance = var_dict[var_id]
            axs_simple = all_axs_simple[costs_id][var_id]
            axs_cumu = all_axs_cumu[costs_id][var_id]

            axs_simple.grid(which="major")
            axs_simple.grid(which="minor", linestyle=":", alpha=0.3)
            axs_cumu.grid(which="major")
            axs_cumu.grid(which="minor", linestyle=":", alpha=0.3)

            for acquisition in acquisitions:
                if acquisition == "ucb" or acquisition == "ts":
                    acq_alias = acquisition + "_es0"
                    virtual_costs_id = 0
                    virtual_var_id = 0
                else:
                    acq_alias = acquisition
                    virtual_costs_id = costs_id
                    virtual_var_id = var_id

                color = color_dict[acquisition]
                all_cost_per_iter_cumusums = []
                all_simple_regrets = []
                all_cumu_regrets = []

                for i, seed in enumerate(seeds):
                    filename = (
                        f"{obj_name}_{acq_alias}_c{virtual_costs_id}"
                        f"_var{virtual_var_id}_C{budget}_seed{seed}"
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

                    all_cost_per_iter_cumusums.append(np.cumsum(cost_per_iter))
                    all_simple_regrets.append(simple_regret)
                    all_cumu_regrets.append(cs_cumu_regret)

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
                acq_name = acq_name_dict[acquisition]

                axs_simple.plot(
                    cost_axis, mean_simple_regrets, label=acq_name, color=color
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
                    f"costs: {costs_alias}, variance: {variance}", size=text_size
                )
                axs_simple.set_xlabel("Budget $C$", size=text_size)
                axs_simple.set_ylabel("Simple regret", size=text_size)
                axs_simple.tick_params(labelsize=tick_size)
                axs_simple.legend(fontsize=text_size - 2, loc=legend_loc)

                axs_cumu.set_title(
                    f"costs: {costs_alias}, variance: {variance}", size=text_size
                )
                axs_cumu.set_xlabel("Budget $C$", size=text_size)
                axs_cumu.set_ylabel("Cumulative regret", size=text_size)
                axs_cumu.tick_params(labelsize=tick_size)
                axs_cumu.legend(fontsize=text_size - 2, loc=legend_loc)

    fig_simple.tight_layout()
    fig_simple.savefig(
        save_dir + f"{obj_name}-simple_regret.pdf",
        dpi=dpi,
        bbox_inches="tight",
        format="pdf",
    )

    fig_cumu.tight_layout()
    fig_cumu.savefig(
        save_dir + f"{obj_name}-cumu_regret.pdf",
        dpi=dpi,
        bbox_inches="tight",
        format="pdf",
    )

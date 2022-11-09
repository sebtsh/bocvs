import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch


def get_cumu_regret(X, obj_func, opt_val):
    sample_regret = opt_val - obj_func(X).squeeze(-1)  # (num_iters, )
    return torch.cumsum(sample_regret, dim=0).cpu().detach().numpy()


def get_simple_regret(X, obj_func, opt_val):
    return (
        opt_val - torch.cummax(obj_func(X).squeeze(-1), dim=0)[0].cpu().detach().numpy()
    )


def plot_regret(
    regret, num_iters, title="", save=False, save_dir="", filename="", show_plot=False
):
    fig, (ax1) = plt.subplots(1, 1)
    fig.suptitle(title, size=12)
    fig.set_size_inches(12, 6)
    fig.set_dpi(200)

    ax1.plot(np.arange(num_iters), regret)
    ax1.axhline(y=0, xmax=num_iters, color="grey", alpha=0.5, linestyle="--")

    fig.tight_layout()
    if save:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir + filename, bbox_inches="tight")
    if show_plot:
        plt.show()

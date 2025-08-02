import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
import torch


# Make Directory
def make_directory(folder_name: str, figure_name: str) -> str:
    # Make directory if not exist
    if figure_name is not None:
        path = f"../{folder_name}/{figure_name}"
    else:
        path = f"../{folder_name}"
    folder_exist = os.path.exists(path)
    if not folder_exist:  # make folder if does not exist
        os.makedirs(path)
    return path


def plot_function(x: torch.tensor, y: torch.tensor):
    fig, ax = plt.subplots(figsize=(8, 6), dpi=120)
    ax.plot(x.numpy(), y.numpy(), label="y = x^2", c="red", linewidth=2.5)
    ax.set_title("Function Approximation", weight="bold", fontsize=16)
    ax.set_xlabel("X-axis", fontsize=14)
    ax.set_ylabel("Y-axis", fontsize=14)
    ax.set_ylim([-1, 5])
    ax.margins(x=0, y=0.1)  # no margins on x and y-axis
    ax.grid(color="blue", linestyle="--", linewidth=1, alpha=0.2)
    spines = ["top", "right", "bottom", "left"]
    for s in spines:
        ax.spines[s].set_visible(False)
    ax.legend()
    fig.tight_layout(pad=0.5)
    plt.show()


# Plot
def plot_model(
    x: np.array,
    y: np.array,
    predicted: np.array,
    title: str = "Linear Model",
    xlim: list = [],
    ylim: list = [],
    blue_legend: str = "Linear Model",
) -> None:
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
    if len(ylim) > 0:
        ax.set_ylim(ylim)
    elif len(xlim) > 0:
        ax.set_xlim(xlim)
    else:
        ax.set_ylim([-1, 4.5])
    ax.plot(x.numpy(), y.numpy(), "r-", linewidth=2.5, label="y=x^2")
    ax.plot(x.numpy(), predicted, "b-", linewidth=2.5, label=blue_legend)
    fig.suptitle(f"{title}", weight="bold", fontsize=16)
    ax.margins(x=0, y=0.1)  # No margins on x and y-axis
    ax.grid(color="blue", linestyle="--", linewidth=1, alpha=0.2)
    ax.set_xlabel("X-axis", fontsize=12)
    ax.set_ylabel("Y-axis", fontsize=12)
    ax.legend(loc="upper right")
    ax.legend(loc="best")
    spines = ["top", "right", "bottom", "left"]
    for s in spines:
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.5)
    plt.show()


def plot_animation(
    x: np.array,
    y: np.array,
    preds: Dict[str, np.array],
    file_name: str = "NonLinear",
    folder_name: str = "non-linear-plots",
    red_legend: str = "y=x^2",
    legend_loc: str = "best",
) -> None:
    for epoch, predicted in preds.items():
        path = make_directory("plots", f"{folder_name}")
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=200)
        ax.set_ylim([-1, 4.5])
        ax.plot(x.numpy(), y.numpy(), "r-", linewidth=2.5, label=red_legend)
        ax.plot(x.numpy(), predicted, "b-", linewidth=2.5, label="Non-Linear Model")
        fig.suptitle(f"Non-Linear Model, epoch={epoch}", weight="bold", fontsize=16)
        ax.margins(x=0, y=0.1)  # No margins on x and y-axis
        ax.grid(color="blue", linestyle="--", linewidth=1, alpha=0.2)
        ax.set_xlabel("X-axis", fontsize=12)
        ax.set_ylabel("Y-axis", fontsize=12)
        ax.legend(loc=legend_loc)
        spines = ["top", "right", "bottom", "left"]
        for s in spines:
            ax.spines[s].set_visible(False)
        fig.tight_layout(pad=0.5)
        plt.savefig(f"./{path}/{file_name}_{epoch}.png", bbox_inches="tight", dpi=200)
        # plt.show()
        plt.close()

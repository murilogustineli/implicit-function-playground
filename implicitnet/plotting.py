import os
from typing import Dict

import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms


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


def plot_function(
    x: torch.tensor,
    y: torch.tensor,
    ylim: list = [-1, 5],
    function_name: str = "y = x^2",
    figsize: tuple = (8, 6),
    dpi: int = 120,
):
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

    # calculate scaling factor based on figure area relative to baseline (8,6)
    baseline_area = 8 * 6
    current_area = figsize[0] * figsize[1]
    scaling_factor = (current_area / baseline_area) ** 0.5

    ax.plot(
        x.numpy(),
        y.numpy(),
        label=function_name,
        c="red",
        linewidth=2.5 * scaling_factor,
        linestyle="--",
    )
    ax.set_title("Function Approximation", weight="bold", fontsize=16 * scaling_factor)
    ax.set_xlabel("X-axis", fontsize=14 * scaling_factor)
    ax.set_ylabel("Y-axis", fontsize=14 * scaling_factor)
    ax.tick_params(axis="both", which="major", labelsize=12 * scaling_factor)
    ax.set_ylim(ylim)
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
    ylim: list = [-1, 5],
    function_name: str = "y = x^2",
    figsize: tuple = (8, 6),
    linewidth: float = 2.5,
    dpi: int = 120,
) -> None:
    # create figure and axis
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # calculate scaling factor based on figure area relative to baseline (8,6)
    baseline_area = 8 * 6
    current_area = figsize[0] * figsize[1]
    scaling_factor = (current_area / baseline_area) ** 0.5

    # plot function and model prediction
    ax.plot(
        x.numpy(),
        y.numpy(),
        c="red",
        linewidth=linewidth,
        label=function_name,
        linestyle="--",
    )
    ax.plot(x.numpy(), predicted, c="blue", linewidth=linewidth, label=title)
    ax.set_title(f"{title}", weight="bold", fontsize=16 * scaling_factor)
    ax.margins(x=0, y=0.1)  # No margins on x and y-axis
    ax.grid(color="blue", linestyle="--", linewidth=1, alpha=0.2)
    ax.set_xlabel("X-axis", fontsize=14 * scaling_factor)
    ax.set_ylabel("Y-axis", fontsize=14 * scaling_factor)
    ax.tick_params(axis="both", which="major", labelsize=12 * scaling_factor)
    ax.set_ylim(ylim)
    ax.legend(loc="upper right")
    ax.legend(loc="best")
    spines = ["top", "right", "bottom", "left"]
    for s in spines:
        ax.spines[s].set_visible(False)
    fig.tight_layout(pad=0.5)
    plt.show()


def plot_model_predictions(
    x: np.array,
    y: np.array,
    predictions: list,
    title: str = "Linear Model",
    ylim: list = [-1, 5],
    function_name: str = "y = x^2",
    figsize: tuple = (8, 6),
    rows_cols: tuple = (4, 1),
    dpi: int = 120,
):
    # create subplots based on rows and columns
    fig, ax = plt.subplots(rows_cols[0], rows_cols[1], figsize=figsize, dpi=dpi)

    # get predictions for different epochs dynamically
    num_preds = len(predictions)
    num_plots = rows_cols[0] * rows_cols[1]
    interval = num_preds // 8
    intervals = list(range(0, num_preds, interval))
    pred_intervals = [intervals[i] for i in range(num_plots - 1)]
    # ensure the last interval is included
    pred_intervals += [num_preds - (num_preds % 100)]
    curves = [predictions[i] for i in pred_intervals]

    # calculate scaling factor based on individual subplot area relative to baseline (8,6)
    baseline_area = 8 * 6
    subplot_area = (figsize[0] * figsize[1]) / (rows_cols[0] * rows_cols[1])
    scaling_factor = (subplot_area / baseline_area) ** 0.5

    for i, curve in enumerate(curves):
        ax[i].plot(
            x.numpy(),
            y.numpy(),
            c="red",
            linewidth=3 * scaling_factor,
            linestyle="--",
            label=function_name,
        )
        ax[i].plot(
            x.numpy(),
            curve,
            c="blue",
            linewidth=3 * scaling_factor,
            linestyle="-",
            label=f"Model Prediction at Epoch {pred_intervals[i]}",
        )
        ax[i].set_title(
            f"{title} - Epoch {pred_intervals[i]}",
            weight="bold",
            fontsize=16 * scaling_factor,
        )
        ax[i].set_ylim(ylim)
        ax[i].set_xlabel("X-axis", fontsize=14 * scaling_factor)
        ax[i].set_ylabel("Y-axis", fontsize=14 * scaling_factor)
        ax[i].tick_params(axis="both", which="major", labelsize=12 * scaling_factor)
        ax[i].margins(x=0, y=0.1)  # No margins on x and y-axis
        ax[i].grid(color="blue", linestyle="--", linewidth=1, alpha=0.2)
        for spine in ax[i].spines.values():
            spine.set_visible(False)
        ax[i].legend(loc="upper right", fontsize=12 * scaling_factor)

    fig.tight_layout(pad=0.5)
    plt.show()


def plot_animation(
    x: np.array,
    y: np.array,
    preds: Dict[str, np.array],
    file_name: str = "linear",
    folder_name: str = "linear_plots",
    model_name: str = "Linear Model",
    legend_loc: str = "best",
    ylim: list = [-1, 5],
    function_name: str = "y=x^2",
    figsize: tuple = (8, 6),
    linewidth: float = 2.5,
    dpi: int = 200,
) -> None:
    # calculate scaling factor based on figure area relative to baseline (8,6)
    baseline_area = 8 * 6
    current_area = figsize[0] * figsize[1]
    scaling_factor = (current_area / baseline_area) ** 0.5

    # plot each epoch
    for epoch, predicted in preds.items():
        path = make_directory("plots", f"{folder_name}")
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.plot(
            x.numpy(),
            y.numpy(),
            c="red",
            linewidth=linewidth,
            label=function_name,
            linestyle="--",
        )
        ax.plot(x.numpy(), predicted, c="blue", linewidth=linewidth, label=model_name)
        ax.set_title(
            f"{model_name}, epoch={epoch}", weight="bold", fontsize=16 * scaling_factor
        )
        ax.margins(x=0, y=0.1)  # No margins on x and y-axis
        ax.grid(color="blue", linestyle="--", linewidth=1, alpha=0.2)
        ax.set_xlabel("X-axis", fontsize=14 * scaling_factor)
        ax.set_ylabel("Y-axis", fontsize=14 * scaling_factor)
        ax.tick_params(axis="both", which="major", labelsize=12 * scaling_factor)
        ax.set_ylim(ylim)
        ax.legend(loc=legend_loc)
        spines = ["top", "right", "bottom", "left"]
        for s in spines:
            ax.spines[s].set_visible(False)
        fig.tight_layout(pad=0.5)
        plt.savefig(f"./{path}/{file_name}_{epoch}.png", bbox_inches="tight", dpi=200)
        # plt.show()
        plt.close()


def create_gif(
    folder_name: str = "linear_plots",
    file_name: str = "linear",
    n_iters: list = None,
    duration: int = 64,
) -> None:
    # get paths and create directories
    path = os.path.abspath(os.path.join(os.getcwd(), ".."))
    plots_dir = os.path.join(path, "plots", folder_name)
    animation_dir = os.path.join(path, "animations")
    if not os.path.exists(animation_dir):
        os.makedirs(animation_dir)

    # create animation
    frames = []
    for epoch in n_iters:
        image = imageio.v2.imread(f"{plots_dir}/{file_name}_{epoch}.png")
        frames.append(image)

    imageio.mimsave(
        f"{animation_dir}/{file_name}_animation.gif",  # output gif
        frames,  # array of input frames
        duration=duration,  # optional: frames per second
        loop=0,  # loop the gif
    )


def plot_image_predictions(
    predictions: list,
    height: int = 28,
    width: int = 28,
    rows_cols: tuple = (1, 4),
    cmap: str = "plasma",
    dpi: int = 120,
    title_prefix: str = "Prediction",
    selected_epochs: list = None,
):
    num_plots = rows_cols[0] * rows_cols[1]
    num_preds = len(predictions)

    # select evenly spaced snapshots
    if not selected_epochs:
        interval = max(1, num_preds // num_plots)
        selected_epochs = list(range(0, num_preds, interval))[: num_plots - 1]
        selected_epochs.append(num_preds - (num_preds % 100))
        selected_preds = [predictions[i] for i in selected_epochs]
    else:
        selected_preds = [predictions[i] for i in selected_epochs]

    # create plot grid
    fig, ax = plt.subplots(
        rows_cols[0],
        rows_cols[1],
        figsize=(4 * rows_cols[1], 4 * rows_cols[0]),
        dpi=dpi,
    )
    ax = ax.flatten()

    for i, pred in enumerate(selected_preds):
        pred_image = np.clip(pred.reshape(height, width), 0, 1)
        ax[i].imshow(pred_image, cmap=cmap, vmin=0, vmax=1)
        ax[i].set_title(
            f"{title_prefix} - Epoch {selected_epochs[i]}", fontsize=12, weight="bold"
        )
        ax[i].axis("off")

    fig.tight_layout()
    plt.show()


def load_preprocess_image(image_path, target_size: int = 128):
    # load and crop to centered square
    img = Image.open(image_path).convert("RGB")
    min_dim = min(img.size)
    width, height = img.size
    left = (width - min_dim) / 2
    top = (height - min_dim) / 2
    right = (width + min_dim) / 2
    bottom = (height + min_dim) / 2
    image = img.crop((left, top, right, bottom))

    # convert to grayscale and resize
    transform = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((target_size, target_size)),
            transforms.ToTensor(),  # converts to [0, 1] and shape [1, H, W]
        ]
    )

    return transform(image).squeeze(0)  # shape: [H, W]

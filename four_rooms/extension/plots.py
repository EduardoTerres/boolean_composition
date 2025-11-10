import random

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

import deepdish as dd
from numpy import save

from four_rooms.extension.plot_utils import (
    plot_returns_all_num_goals,
    plot_time_taken_all_num_goals,
)

random.seed(42)

def plot(
    num_rooms,
    tasks,
    save_name = None,
):
    tasks = ["\n".join(str(goal) for goal in task) for task in tasks]
    plt.ylim(-0.5, 2)
    rc_ = {
        "figure.figsize": (30, 10),
        "axes.labelsize": 30,
        "font.size": 30,
        "legend.fontsize": 20,
        "axes.titlesize": 30,
    }
    sns.set(rc=rc_, style="darkgrid", font_scale=1.8)
    rc("text", usetex=False)

    n = 2

    data = dd.io.load(f"exps_data_extension/exp2_all_returns_{num_rooms}.h5")
    data = [data[t] / 10 for t in range(len(data))]

    types = [
        "Sparse rewards and Same absorbing set",
        "Dense rewards and Same absorbing set",
        "Sparse rewards and Different absorbing set",
        "Dense rewards and Different absorbing set",
    ]

    data = pd.DataFrame(
        [[data[0][i, t] for t in range(n, len(tasks))] + [types[0]] for i in range(len(data[0]))]
        + [[data[1][i, t] for t in range(n, len(tasks))] + [types[1]] for i in range(len(data[1]))]
        + [[data[2][i, t] for t in range(n, len(tasks))] + [types[2]] for i in range(len(data[2]))]
        + [[data[3][i, t] for t in range(n, len(tasks))] + [types[3]] for i in range(len(data[3]))],
        columns=tasks[n:] + ["Domain"],
    )
    data = pd.melt(data, "Domain", var_name="Tasks", value_name="Average Returns")

    fig, ax = plt.subplots()
    ax = sns.boxplot(
        x="Tasks",
        y="Average Returns",
        hue="Domain",
        data=data,
        linewidth=3,
        showfliers=False,
    )
    if save_name is None:
        save_name = f"four_rooms/extension/figures/exp2_output_{num_rooms}.png"
    fig.savefig(save_name, bbox_inches="tight")
    print(f"Figure saved to {save_name}")

# if __name__ == "__main__":
#     from four_rooms.config import Config_4
#     print(Config_4["Tasks"])
#     plot(num_rooms=4, tasks=Config_4["Tasks"])

def plot_time_taken_all():
    time_taken_4 = dd.io.load("exps_data_extension/composed_time_taken_4.h5")
    time_taken_8 = dd.io.load("exps_data_extension/composed_time_taken_8.h5")
    time_taken_16 = dd.io.load("exps_data_extension/composed_time_taken_16.h5")
    time_taken = {4: time_taken_4, 8: time_taken_8, 16: time_taken_16}
    plot_time_taken_all_num_goals(time_taken, save_name="four_rooms/extension/figures/time_taken_all_num_goals.png")

def plot_returns_all():
    returns_4 = dd.io.load("exps_data_extension/composed_returns_4.h5")
    returns_8 = dd.io.load("exps_data_extension/composed_returns_8.h5")
    returns_16 = dd.io.load("exps_data_extension/composed_returns_16.h5")
    returns = {4: returns_4, 8: returns_8, 16: returns_16}
    plot_returns_all_num_goals(returns, save_name="four_rooms/extension/figures/returns_all_num_goals.png")

if __name__ == "__main__":
    plot_returns_all()

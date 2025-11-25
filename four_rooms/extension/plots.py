import random

from numpy import mean
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc

import deepdish as dd

from four_rooms.extension.plot_utils import (
    plot_extended_q_value,
    plot_returns_all_num_goals,
    plot_returns_optimality_all_num_goals,
    plot_time_taken_all_num_goals,
    plot_num_value_functions_learned,
)

from four_rooms.config import Config_4, Config_8, Config_16


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

def generate_plot_time_taken_all_num_goals():
    num_runs = 1000
    maxiter = 50

    time_taken_4 = dd.io.load(f"exps_data_extension/composed_time_taken_4_{maxiter}_{num_runs}.h5")
    time_taken_8 = dd.io.load(f"exps_data_extension/composed_time_taken_8_{maxiter}_{num_runs}.h5")
    time_taken_16 = dd.io.load(f"exps_data_extension/composed_time_taken_16_{maxiter}_{num_runs}.h5")
    time_taken = {4: time_taken_4, 8: time_taken_8, 16: time_taken_16}
    plot_time_taken_all_num_goals(time_taken, save_name=f"four_rooms/extension/figures/time_taken_all_num_goals_{maxiter}_{num_runs}.png")

def generate_plot_returns_all_num_goals():
    maxiter = 5_000
    num_runs = 10_000

    returns_4 = dd.io.load(f"exps_data_extension/composed_returns_4_{maxiter}_{num_runs}.h5")
    returns_8 = dd.io.load(f"exps_data_extension/composed_returns_8_{maxiter}_{num_runs}.h5")
    returns_16 = dd.io.load(f"exps_data_extension/composed_returns_16_{maxiter}_{num_runs}.h5")
    returns = {4: returns_4, 8: returns_8, 16: returns_16}
    plot_returns_all_num_goals(returns, save_name=f"four_rooms/extension/figures/returns_all_num_goals_{maxiter}_{num_runs}.png")

def generate_plot_num_value_functions_learned():
    save_name = "four_rooms/extension/figures/num_value_functions_learned.png"
    plot_num_value_functions_learned(save_name)

def generate_plot_extended_q_value(on_off: str):
    num_rooms = 16
    maxiter = 1000

    configs = {
        4: Config_4,
        8: Config_8,
        16: Config_16,
    }
    config = configs[num_rooms]

    terminal_states = config["T_states"]
    goals = config["Goals"]

    terminal_states = Config_8["T_states"]

    learned_EQ = dd.io.load(f"exps_data_extension/learned_EQ_{on_off}_{num_rooms}_{maxiter}.h5")

    save_name = f"four_rooms/extension/figures/extended_q_value_{num_rooms}_{maxiter}_{on_off}"
    plot_extended_q_value(learned_EQ, goals, terminal_states, num_rooms, save_name)

def generate_plot_returns_optimality():

    returns_per_maxiter = {}
    optimal_return = {}
    for num_rooms in [4, 8, 16]:
        returns_per_maxiter[num_rooms] = dd.io.load(f"exps_data_extension/convergence_returns_{num_rooms}.h5")
        optimal_returns = dd.io.load(f"exps_data_extension/composed_returns_{num_rooms}_1000_10000.h5")
        optimal_return[num_rooms] = mean([mean(optimal_returns[task]["optimal"]) for task in optimal_returns])

    plot_returns_optimality_all_num_goals(
        returns_per_maxiter,
        optimal_return,
        save_name="four_rooms/extension/figures/returns_optimality.png",
    )

if __name__ == "__main__":
    # generate_plot_time_taken_all_num_goals()
    generate_plot_returns_all_num_goals()
    # generate_plot_returns_optimality()
    # generate_plot_num_value_functions_learned()
    # generate_plot_extended_q_value(on_off="off")
    # generate_plot_extended_q_value(on_off="on")

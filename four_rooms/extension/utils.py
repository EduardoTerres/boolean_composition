import deepdish as dd
from numpy import save
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from matplotlib import rc

def plot(
    num_rooms,
    save_name = None,
):
    tasks = [
        r"${M_{\emptyset}}$",
        r"${M_{\mathcal{U}}}$",
        r"${M_{T}}\wedge{M_{L}}$",
        r"${M_{T}}\wedge\neg{M_{L}}$",
        r"${M_{L}}\wedge\neg{M_{T}}$",
        r"${M_{T}}\bar{\vee}{M_{L}}$",
        r"${M_{T}}$",
        r"$\neg {M_{T}}$",
        r"${M_{L}}$",
        r"$\neg {M_{L}}$",
        r"${M_{T}}\vee{M_{L}}$",
        r"${M_{T}}\vee\neg{M_{L}}$",
        r"${M_{L}}\vee\neg{M_{T}}$",
        r"${M_{T}}\bar{\wedge}{M_{L}}$",
        r"$\neg({M_{T}} \veebar {M_{L}})$",
        r"${M_{T}} \veebar {M_{L}}$",
    ]

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
        [[data[0][i, t] for t in range(n, 16)] + [types[0]] for i in range(len(data[0]))]
        + [[data[1][i, t] for t in range(n, 16)] + [types[1]] for i in range(len(data[1]))]
        + [[data[2][i, t] for t in range(n, 16)] + [types[2]] for i in range(len(data[2]))]
        + [[data[3][i, t] for t in range(n, 16)] + [types[3]] for i in range(len(data[3]))],
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
    plt.show()
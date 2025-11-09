from four_rooms.GridWorld import GridWorld
from four_rooms.library import (
    EQ_P,
    EQ_V,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def plot_composed_EQs(composed_EQs, goals, terminal_states, num_rooms):
    """Plot composed EQs for all tasks individually."""
    tasks = list(composed_EQs.keys())
    
    for row_idx, task in enumerate(tasks):
        env = GridWorld(
            MAP="MAP_" + str(num_rooms),
            goals=goals,
            T_states=terminal_states,
        )
        
        # Render onoff EQ and move axes to main figure
        on_off_fig = env.render(P=EQ_P(composed_EQs[task]["onoff"]), V=EQ_V(composed_EQs[task]["onoff"]))
        boolean_fig = env.render(P=EQ_P(composed_EQs[task]["boolean"]), V=EQ_V(composed_EQs[task]["boolean"]))

        # Save the figs to pngs
        on_off_fig.savefig(f"four_rooms/extension/figures_comparison/on_off_task_{row_idx + 1}_rooms_{num_rooms}.png")
        boolean_fig.savefig(f"four_rooms/extension/figures_comparison/boolean_task_{row_idx + 1}_rooms_{num_rooms}.png")

        # Close the figs
        plt.close(on_off_fig)
        plt.close(boolean_fig)


def plot_returns(returns: dict[tuple[int, int], dict[str, list[float]]], num_rooms: int, save_name: str = None):
    """ Plot returns for all tasks, comparing onoff and boolean methods side by side."""
    tasks = ["\n".join(str(g) for g in task) for task in returns.keys()]
    data = pd.DataFrame([{"Task": task, "Method": method, "Returns": val}
                         for task, vals in zip(tasks, returns.values())
                         for method, returns_list in vals.items()
                         for val in returns_list])
    plt.figure(figsize=(16, 6))  # Make the figure bigger
    sns.set_context("notebook", font_scale=0.8)  # Make the words smaller
    ax = sns.boxplot(x="Task", y="Returns", hue="Method", data=data)
    ax.set_xlabel("Task", fontsize=20)
    ax.set_ylabel("Returns", fontsize=20)
    ax.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()


def plot_time_taken(time_taken: dict[str, list[float]], num_rooms: int, save_name: str = None):
    """ Plot returns for all tasks, comparing onoff and boolean methods side by side."""
    tasks = ["\n".join(str(g) for g in task) for task in time_taken.keys()]
    data = pd.DataFrame([{"Task": task, "Method": method, "Returns": val}
                         for task, vals in zip(tasks, time_taken.values())
                         for method, returns_list in vals.items()
                         for val in returns_list])
    plt.figure(figsize=(6, 6))  # Make the figure bigger
    sns.set_context("notebook", font_scale=0.8)  # Make the words smaller
    ax = sns.boxplot(x="Task", y="Returns", hue="Method", data=data)
    ax.set_xlabel("Task", fontsize=20)
    ax.set_ylabel("Returns", fontsize=20)
    ax.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(save_name)
    plt.show()


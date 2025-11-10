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


def plot_time_taken(time_taken: dict[str, list[float]], num_rooms: int, save_name: str = None):
    """ Plot returns for all tasks, comparing onoff and boolean methods side by side."""
    tasks = ["\n".join(str(g) for g in task) for task in time_taken.keys()]
    data = pd.DataFrame([{"Task": task, "Method": method, "Returns": val}
                         for task, vals in zip(tasks, time_taken.values())
                         for method, returns_list in vals.items()
                         for val in returns_list])
    plt.figure(figsize=(12, 12))  # Make the figure bigger
    sns.set_context("notebook", font_scale=0.8)  # Make the words smaller
    ax = sns.boxplot(x="Task", y="Returns", hue="Method", data=data)
    # Do not plot the x ticks labels
    ax.set_ylabel("Time taken for composition", fontsize=20)
    ax.set_xlabel("All tasks", fontsize=20)
    ax.legend(fontsize=20)
    ax.set_xticklabels([])  # Remove x tick labels
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(save_name)


def plot_time_taken_all_num_rooms(time_taken: dict[int, dict[str, list[float]]], save_name: str = None):
    """ Plot time taken for all number of rooms (log scale on y-axis)."""
    num_rooms = sorted(time_taken.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = {r: i for i, r in enumerate(num_rooms)}
    
    colors = plt.cm.tab10.colors[:2]
    for idx, method in enumerate(["onoff", "boolean"]):
        data = [time_taken[r][method] for r in num_rooms]
        bp = ax.boxplot(data, positions=[positions[r] for r in num_rooms], widths=0.3, patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(colors[idx])
            patch.set_alpha(0.7)
        # Set the color of the median lines
        for median in bp['medians']:
            median.set_color(colors[idx])
        means = [sum(vals) / len(vals) for vals in data]
        ax.plot([positions[r] for r in num_rooms], means, 'o-', label=method, linewidth=2, color=colors[idx])

    ax.set_xticks(range(len(num_rooms)))
    ax.set_xticklabels(num_rooms)
    ax.set_xlabel("Number of rooms", fontsize=16)
    ax.set_ylabel("Time taken", fontsize=16)
    ax.set_yscale('log')
    ax.legend(fontsize=14)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)

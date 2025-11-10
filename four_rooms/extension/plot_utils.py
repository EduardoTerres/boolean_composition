import random
from collections import defaultdict
from four_rooms.GridWorld import GridWorld
from four_rooms.library import (
    EQ_P,
    EQ_V,
)
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from matplotlib import rc

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


def plot_returns(returns: dict[tuple[int, int], dict[str, list[float]]], save_name: str = None):
    """ Plot returns for all tasks, comparing onoff and boolean methods side by side."""
    tasks = ["\n".join(str(g) for g in task) for task in returns.keys()]
    data = pd.DataFrame([{"Task": task, "Method": method, "Returns": val}
                         for task, vals in zip(tasks, returns.values())
                         for method, returns_list in vals.items()
                         for val in returns_list])
    plt.figure(figsize=(16, 6))
    sns.set_context("notebook", font_scale=0.8)
    ax = sns.boxplot(x="Task", y="Returns", hue="Method", data=data)
    ax.set_xlabel("Task", fontsize=20)
    ax.set_ylabel("Returns", fontsize=20)
    ax.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.tight_layout()
    plt.savefig(save_name)


def plot_returns_all_num_goals(
    returns: dict[int, dict[tuple[int, int], dict[str, list[float]]]],
    save_name: str = None,
):
    """ Plot returns for all tasks, comparing onoff and boolean methods side by side.
    
    Args:
        returns: Dictionary mapping num_rooms to tasks to their returns
        save_name: Path to save the figure
    """
    rc("text", usetex=True)
    # Number of tasks to show for each number of goals
    # The keys of the outer dictionary are the number of rooms
    # The keys of the inner dictionary are the number of goals
    # The values are the number of tasks to show for each number of goals
    shown_tasks = {
        4: [1, 2, 3],
        8: [1, 3, 5, 7],
        16: [1, 5, 9, 13],
    }

    flattened_returns = {}
    # Proportional sample of the tasks for each number of rooms
    for num_rooms, task_returns_dict in returns.items():
        tasks_by_length = defaultdict(list)
        for task in task_returns_dict.keys():
            tasks_by_length[len(task)].append(task)
        
        for length in shown_tasks[num_rooms]:
            tasks_of_length_returns = {
                "onoff": [],
                "boolean": [],
            }
            for task in tasks_by_length[length]:
                tasks_of_length_returns["onoff"].extend(returns[num_rooms][task]["onoff"])
                tasks_of_length_returns["boolean"].extend(returns[num_rooms][task]["boolean"])

            flattened_returns[(num_rooms, length)] = tasks_of_length_returns

    print(
        "Sampled tasks:",
        ", ".join(
            f"{len(returns[num_rooms].keys())} tasks for {num_rooms} rooms"
            for num_rooms in [4, 8, 16]
        ),
    )
    
    tasks = [f"{task[1]}-{task[0]}" for task in flattened_returns.keys()]
    data = pd.DataFrame([{"Task": task, "Method": method, "Returns": val}
                         for task, vals in zip(tasks, flattened_returns.values())
                         for method, returns_list in vals.items()
                         for val in returns_list])
    plt.figure(figsize=(20, 10))
    sns.set_context("notebook", font_scale=0.8)
    ax = sns.boxplot(x="Task", y="Returns", hue="Method", data=data)
    
    # Add vertical lines to separate rooms
    tasks_per_room = [len(shown_tasks[r]) for r in [4, 8, 16]]
    pos = 0
    for n in tasks_per_room[:-1]:
        pos += n
        ax.axvline(pos - 0.5, color='black', linestyle='-', linewidth=1)
    
    # Add text labels below each room block
    pos = 0
    for r, n in zip([4, 8, 16], tasks_per_room):
        ax.text(pos + (n - 1) / 2, -0.14, f"{r} rooms", ha='center', transform=ax.get_xaxis_transform(), fontsize=28)
        pos += n
    
    ax.set_xlabel("Number of goals in task", fontsize=26)
    ax.set_ylabel("Returns", fontsize=26)
    ax.legend(fontsize=26)

    # Change labels
    ax.set_xticklabels([tick.get_text().split('-')[0] for tick in ax.get_xticklabels()], fontsize=26)
    plt.yticks(fontsize=26)
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


def plot_time_taken_all_num_goals(time_taken: dict[int, dict[str, list[float]]], save_name: str = None):
    """ Plot time taken for all number of goals (log scale on y-axis)."""
    rc("text", usetex=True)
    num_rooms = sorted(time_taken.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = {r: i for i, r in enumerate(num_rooms)}
    
    colors = plt.cm.tab10.colors[:2]
    for idx, method in enumerate(["onoff", "boolean"]):
        data = [[1000 * t for t in time_taken[r][method]] for r in num_rooms]
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
    ax.set_xticklabels(num_rooms, fontsize=16)
    ax.set_xlabel("Number of rooms", fontsize=16)
    ax.set_ylabel("Time (ms)", fontsize=16)
    ax.set_yticklabels(ax.get_yticks(), fontsize=16)
    ax.set_yscale('log')
    ax.legend(fontsize=14)
    plt.tight_layout()
    if save_name:
        plt.savefig(save_name)

import deepdish as dd
import numpy as np
from tqdm import tqdm

from four_rooms.config import (
    T_states_4,
    Bases_4,
    Tasks_4,
    Goals_4,
    T_states_16,
    Tasks_16,
    Goals_16,
)

def equal_on_shared_goals(
    task_to_EQs,
    tol=1e-9
):
    """Check that the state-action slices of all tasks that contain a shared goal are the same.

    This function compares the state-action slices of all tasks that contain a shared goal among each other.
    Each match signifies a match of between two tasks on one of their shared goals.

    Args:
        EQs: List of EQs learned for the base tasks.
        base_tasks: List of tasks where each task is represented as a list of goals. E.g., [(3, 3), (3, 9)] means
                    the task is to reach the goals (3, 3) and (3, 9).
        tol: Tolerance for checking equality.
    """
    stats = {
        "mismatch_count": 0,
        "match_count": 0,
    }        
    # Compare all tasks containing goal among each other
    for idx_i, (task_i, EQ_i) in enumerate(task_to_EQs.items()):
        for idx_j, (task_j, EQ_j) in enumerate(task_to_EQs.items()):
            if idx_i <= idx_j:
                continue

            # Obtain intersection goals
            intersection_goals = set(task_i).intersection(set(task_j))
            if len(intersection_goals) == 0:
                continue

            states = list(EQ_i.keys())

            for goal in intersection_goals:
                matches = [
                    np.max(np.abs(EQ_i[state][str([goal, goal])] - EQ_j[state][str([goal, goal])])) < tol
                    for state in states
                ]
                stats["mismatch_count" if not all(matches) else "match_count"] += 1
    
    # Print the counts after all states have been checked
    if stats["mismatch_count"] > 0:
        print(f"❌ {stats['mismatch_count']}/{stats['mismatch_count'] + stats['match_count']} goal slices mismatch.")
    if stats["match_count"] > 0:
        print(f"✅ {stats['match_count']}/{stats['mismatch_count'] + stats['match_count']} goal slices match.")

def equal_on_shared_goals_invidx(
    task_to_EQs,
    tasks,
    goals,
    tol=1e-9
):
    """Check that the state-action slices of all tasks that contain a shared goal are the same.

    This function creates an inverted index of goals and tasks that contain that goal, and then
    compares the state-action slices of all tasks that contain a shared goal among each other.
    Each match signifies a match of between two tasks on one of their shared goals.

    Args:
        task_to_EQs: Dictionary mapping tasks to EQs learned for those tasks
        goals: List of goals
        tol: Tolerance for checking equality

    Returns:
        None
    """
    # Create inverted index of goals and tasks that contain that goal
    inverted_goal_index = {}
    for goal in goals:
        inverted_goal_index[goal] = [task for task in tasks if goal in task]
    stats = {
        "mismatch_count": 0,
        "match_count": 0,
    }
    for goal in tqdm(goals, desc="Checking shared goals"):
        tasks_with_goal = inverted_goal_index[goal]
        if len(tasks_with_goal) == 0:
            continue
        
        # Compare all tasks containing goal among each other
        for idx_i, task_i in enumerate(tasks_with_goal):
            for idx_j, task_j in enumerate(tasks_with_goal):
                if idx_i <= idx_j:
                    continue
                EQs_i = task_to_EQs[tuple(task_i)]
                EQs_j = task_to_EQs[tuple(task_j)]

                matches = [
                    np.max(np.abs(EQs_i[state][str([goal, goal])] - EQs_j[state][str([goal, goal])])) < tol
                    for state in EQs_i.keys()
                ]
                stats["mismatch_count" if not all(matches) else "match_count"] += 1

    # Print the counts after all states have been checked
    if stats["mismatch_count"] > 0:
        print(f"❌ {stats['mismatch_count']} out of {stats['mismatch_count'] + stats['match_count']} mismatches.")
    if stats["match_count"] > 0:
        print(f"✅ {stats['match_count']} out of {stats['mismatch_count'] + stats['match_count']} matches.")

# ------------------------------------------------------------
# Test 1 - Base tasks from original experiments of 4 rooms
# ------------------------------------------------------------
EQs_A = dd.io.load("exps_data/exp3_base_tasks_A_0.h5")
EQs_B = dd.io.load("exps_data/exp3_base_tasks_B_0.h5")

print("Test base tasks from original experiments of 4 rooms", end=" ")
equal_on_shared_goals(task_to_EQs={tuple(Bases_4[0]): EQs_A, tuple(Bases_4[1]): EQs_B})

# ------------------------------------------------------------
# Test 2 - Randomly sampled tasks trained from scratch
# ------------------------------------------------------------
def parse_filename(fname, print_params=False):
    """Parse filename to extract experiment parameters.
    
    Format: exp1_<number_of_rooms>_<number_of_goals>_<number_of_tasks_learned>_<type_of_environment>_<maxiter>.h5
    """
    parts = fname.split('/')[-1].replace('.h5', '').split('_')
    if print_params:
        print(f"Loading: rooms={parts[1]}, goals={parts[2]}, tasks={parts[3]}, env={parts[4]}, maxiter={parts[5]}", end=" ")
    return {
        'number_of_rooms': parts[1],
        'number_of_goals': parts[2],
        'number_of_tasks_learned': parts[3],
        'type_of_environment': parts[4],
        'maxiter': parts[5]
    }

for num_rooms in [4, 9, 16]:
    fname = f"exps_data_extension/exp1_{num_rooms}_3_50_0_2000.h5"
    params = parse_filename(fname, print_params=True)
    task_to_EQs = dd.io.load(fname)
    equal_on_shared_goals(task_to_EQs=task_to_EQs)



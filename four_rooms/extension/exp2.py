from collections import defaultdict
import random
import numpy as np
import deepdish as dd
from four_rooms.GridWorld import GridWorld
from tqdm import tqdm
from four_rooms.library import (
    EQ_P,
    AND,
    OR,
    NOT,
    Goal_Oriented_Q_learning,
)
from four_rooms.config import (
    Config_4,
    Config_9,
    Config_16,
)

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def get_random_partition(Goals):
    """Returns the set partitioned in two but with all the elements."""
    random.shuffle(Goals)
    partition_point = random.randint(1, len(Goals) - 1)
    return Goals[:partition_point], Goals[partition_point:]


def evaluate(goals, EQ):
    env = GridWorld(goals=goals, T_states=T_states)
    policy = EQ_P(EQ)
    state = env.reset()
    done = False
    t = 0
    G = 0
    while not done and t < 100:
        action = policy[state]
        state_, reward, done, _ = env.step(action)
        state = state_
        G += reward
        t += 1
    return G


def order_EQs(EQs, Partition):
    """
    Reorder EQs so that the first EQ contains the Q-values for all the goals being undesired (indx 0),
    and the second EQ contains the Q-values for all the goals being desired (indx 1).

    For each goal in Goals, this function checks which partition (Partition[0] or Partition[1]) the goal belongs to,
    and ensures that the Q-values for that goal are placed in the corresponding EQ dictionary in EQs_ordered.
    This is used to separate Q-values for desired and undesired goals for further boolean composition.

    Args:
        EQs (list): List of two EQ dictionaries. Each dictionary maps states to goal-specific Q-values.
        Partition (tuple): Tuple of two lists of goals. Partition[0] contains one subset of goals, Partition[1] the other.

    Returns:
        list: List of two EQ dictionaries, reordered so that each contains Q-values only for the goals in its partition.
        The first EQ is for undesired; the second for desired.
    """
    EQs_ordered = EQs.copy()  # [EQ_undesired, EQ_desired]

    # Append the desired tasks in the first position, and the undesired in the second
    for idx, EQ in enumerate(EQs):
        for goal in Goals:
            # Position state-action slice in index 0 if undesired, 1 if desired
            for state in EQ.keys():
                desired = int(goal in Partition[idx])  # 0 if undesired, 1 if desired
                EQs_ordered[desired][state][str([goal, goal])] = EQs[desired][state][str([goal, goal])]
    return EQs_ordered


# Convert defaultdict objects to regular dictionaries to avoid pickling issues
def convert_defaultdict_to_dict(obj):
    if isinstance(obj, defaultdict):
        return {key: convert_defaultdict_to_dict(value) for key, value in obj.items()}
    elif isinstance(obj, dict):
        return {key: convert_defaultdict_to_dict(value) for key, value in obj.items()}
    else:
        return obj


def build_EQ(task, Goals, EQ_on, EQ_off):
    """Compose an EQ for a specific task by combining Q-values from EQ_on (for task goals) and EQ_off (for non-task goals)."""
    EQ = EQ_on.copy()
    for state in EQ.keys():
        for goal in task:  # Goal is on
            EQ[state][str([goal, goal])] = EQ_on[state][str([goal, goal])]
        for goal in set(Goals) - set(task):  # Goal is off
            EQ[state][str([goal, goal])] = EQ_off[state][str([goal, goal])]
    return EQ


def get_composed_tasks(Tasks, Goals, EQ_on, EQ_off):
    """Generate composed EQs for all tasks by combining Q-values from EQ_on (desired goals) and EQ_off (undesired goals).
    
    Returns the list of tasks with a one to one correspondence to the composed list.
    """
    EQs = []
    for task in Tasks:
        EQs.append(build_EQ(task, Goals, EQ_on, EQ_off))
    return EQs


# ------------------------------------------------------------
# Experiment
# ------------------------------------------------------------
random.seed(42)

NUM_ROOMS = 4
if NUM_ROOMS == 4:
    Config = Config_4
elif NUM_ROOMS == 9:
    Config = Config_9
elif NUM_ROOMS == 16:
    Config = Config_16
else:
    raise ValueError("Invalid number of rooms")

T_states, Goals, Tasks = Config["T_states"], Config["Goals"], Config["Tasks"]

Partition = get_random_partition(Config["Goals"])
# Bases = [[(3, 3), (3, 9)], [(3, 3), (9, 3)]]
# Partition = Bases
print(f"Partitioned goal state into {Partition[0]} and {Partition[1]}.")

# (Sparse rewards, Same terminal states)
types = [(True, True), (True, False), (False, True), (False, False)]

maxiter = 500
num_runs = 10000

EQs_all = {}
Returns_all = {}

for t in range(len(types)):
    print("type: ", t)

    # Learning universal bounds (min and max tasks)
    # env = GridWorld(goals=T_states, dense_rewards=not types[t][0])
    # EQ_max, _ = Goal_Oriented_Q_learning(env, maxiter=maxiter)

    # env = GridWorld(goals=T_states, goal_reward=-0.1, dense_rewards=not types[t][0])
    # EQ_min, _ = Goal_Oriented_Q_learning(env, maxiter=maxiter)

    # Learning base tasks and doing composed tasks
    EQs = []  # [EQ_desired, EQ_undesired]
    for goals_slice in Partition:
        goals = [[pos, pos] for pos in goals_slice]
        env = GridWorld(
            goals=goals,
            dense_rewards=not types[t][0],
            T_states=T_states if types[t][1] else goals,
        )
        EQ, _ = Goal_Oriented_Q_learning(
            env, maxiter=maxiter, T_states=None if types[t][1] else T_states
        )
        EQs.append(EQ)

    EQs_ordered = order_EQs(EQs, Partition)

    EQ_off, EQ_on = EQs_ordered[0], EQs_ordered[1]

    # EQ_off, EQ_on = EQ_min, EQ_max
    EQs_composed = get_composed_tasks(Tasks, Goals, EQ_on, EQ_off)

    # Save base tasks A and B
    np.object = object  # Hack to avoid error in save
    
    EQs_save = {
        "desired": EQ_on,
        "undesired": EQ_off,
    }
    EQs_all[t] = EQs_save

    data = np.zeros((num_runs, len(Tasks)))
    for i in tqdm(range(num_runs), desc="Runs"):
        for j in range(len(Tasks)):
            goals = [[pos, pos] for pos in Tasks[j]]
            data[i, j] = evaluate(goals, EQs_composed[j])

    Returns_all[t] = data

# Convert all Q objects to regular dictionaries
EQs_all_converted = [convert_defaultdict_to_dict(eq) for eq in EQs_all]

np.object = object  # Hack to avoid error in save
dd.io.save(f"exps_data_extension/exp2_all_EQs_{NUM_ROOMS}.h5", EQs_all_converted)
dd.io.save(f"exps_data_extension/exp2_all_returns_{NUM_ROOMS}.h5", Returns_all)

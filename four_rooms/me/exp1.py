import deepdish as dd
import numpy as np

T_states = [(3, 3), (3, 9), (9, 3), (9, 9)]
Goal_positions = [(3, 3), (3, 9), (9, 3), (9, 9)]
T_states = [[pos, pos] for pos in T_states]

Bases = [[(3, 3), (3, 9)], [(3, 3), (9, 3)]]
Tasks = [
    [],
    [(3, 3), (3, 9), (9, 3), (9, 9)],
    [(3, 3)],
    [(3, 9)],
    [(9, 3)],
    [(9, 9)],
    [(3, 3), (3, 9)],
    [(9, 3), (9, 9)],
    [(3, 3), (9, 3)],
    [(3, 9), (9, 9)],
    [(3, 3), (3, 9), (9, 3)],
    [(3, 3), (3, 9), (9, 9)],
    [(3, 3), (9, 3), (9, 9)],
    [(3, 9), (9, 3), (9, 9)],
    [(3, 3), (9, 9)],
    [(3, 9), (9, 3)],
]

# ------------------------------
# Test 1 - Base tasks
# ------------------------------

intersection = set(Bases[0]).intersection(Bases[1])

EQs_A = dd.io.load("exps_data/exp3_base_tasks_A_0.h5")
EQs_B = dd.io.load("exps_data/exp3_base_tasks_B_0.h5")

intersection_goal = '[(3, 3), (3, 3)]'

tol = 1e-9

for state in EQs_A.keys():
    assert (
        np.max(np.abs(EQs_A[state][intersection_goal] - EQs_B[state][intersection_goal])) < tol
    ), "❌ The learned state-action slice for the intersection goal is not the same for both base tasks."

print("✅ The learned state-action slice for the intersection goal is the same for both base tasks.")

# ------------------------------
# Test 2 - All tasks trained from scratch
# ------------------------------
EQs_all = dd.io.load("exps_data/exp3_EQs_all_tasks0.h5")

# EQs_all is a list of 16 elements, positionally corresponding to the tasks in the Tasks list
task_to_EQs = {str(task): eq for task, eq in zip(Tasks, EQs_all)}

states = list(task_to_EQs[str(Tasks[0])].keys())

# Create inverted index of goals and tasks that contain that goal
inverted_goal_index = {}
for goal in Goal_positions:
    inverted_goal_index[str([goal, goal])] = [task for task in Tasks if goal in task]

mismatch_count = 0
match_count = 0
for goal in T_states:
    print("Testing goal: ", goal)
    for state in states:
        tasks_with_goal = inverted_goal_index[str(goal)]
        if len(tasks_with_goal) == 0:
            continue
        
        # Compare all tasks containing goal among each other
        for task_i in tasks_with_goal:
            for task_j in tasks_with_goal:
                if task_i == task_j:
                    continue

                EQs_i = task_to_EQs[str(task_i)]
                EQs_j = task_to_EQs[str(task_j)]

                # Check the state-action slice
                for state in EQs_A.keys():
                    if np.max(np.abs(EQs_i[state][str(goal)] - EQs_j[state][str(goal)])) < tol:
                        match_count += 1
                    else:
                        mismatch_count += 1

# Print the counts after all states have been checked
if mismatch_count > 0:
    print(f"❌ {mismatch_count} out of {mismatch_count + match_count} mismatches.")
if match_count > 0:
    print(f"✅ {match_count} out of {mismatch_count + match_count} matches.")

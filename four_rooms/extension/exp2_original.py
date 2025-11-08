import numpy as np

# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def get_base_tasks(goals):
    """
    Given a list of goals, this function generates a set of 'base tasks' such that
    each base task is a subset of goals corresponding to a column in the binary encoding
    of the goal indices. This is useful for constructing a basis for goal composition,
    where each goal can be represented as a combination of base tasks.

    Arguments:
        goals: list of goals

    Returns:
        base_tasks: list of base tasks
        composition_rules: dictionary of composition rules for each goal
    """
    num_rows, num_cols = len(goals), int(np.ceil(np.log(len(goals))))
    matrix = [
        [int(b) for b in format(i, f'0{num_cols}b')] for i in range(num_rows)
    ]
    matrix = np.array(matrix)
    print(matrix)

    base_tasks = []
    for k in range(num_cols):
        base_tasks.append([])
        for i, goal in enumerate(goals):
            if matrix[i, k] == 1:
                base_tasks[k].append(goal)

    composition_rules = dict(zip(goals, list(matrix)))

    return base_tasks, composition_rules

if __name__ == "__main__":
    from four_rooms.config import Config_4
    print(Config_4["Goals"])
    tasks = Config_4["Goals"]
    print(get_base_tasks(tasks))


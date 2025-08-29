import numpy as np
import deepdish as dd
from pathlib import Path

# Import path
import sys
sys.path.append(Path(__file__).parent.parent)

from four_rooms.GridWorld import GridWorld
from tqdm import tqdm
from four_rooms.library import EQ_P, Goal_Oriented_Q_learning

T_states = [(3, 3), (3, 9), (9, 3), (9, 9)]
T_states = [[pos, pos] for pos in T_states]

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

# (Sparse rewards, Same terminal states)
types = [(True, True), (True, False), (False, True), (False, False)]

maxiter = 500


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


print("type: (Sparse rewards, Same terminal states)")
t = 0

# Learning universal bounds (min and max tasks)
env = GridWorld(goals=T_states, dense_rewards=not types[t][0])
EQ_max, _ = Goal_Oriented_Q_learning(env, maxiter=maxiter)

env = GridWorld(goals=T_states, goal_reward=-0.1, dense_rewards=not types[t][0])
EQ_min, _ = Goal_Oriented_Q_learning(env, maxiter=maxiter)

EQs_learned = []
# Learning base tasks and doing composed tasks
for task in tqdm(Tasks, desc="Training tasks"):
    goals = [[pos, pos] for pos in task]
    env = GridWorld(
        goals=goals,
        dense_rewards=not types[t][0],
        T_states=T_states if types[t][1] else goals,
    )
    Q, stats1 = Goal_Oriented_Q_learning(
        env, maxiter=maxiter, T_states=None if types[t][1] else T_states
    )
    EQs_learned.append(Q)

np.object = object  # Hack to avoid error in save
dd.io.save("exps_data/exp3_EQs_all_tasks" + str(t) + ".h5", EQs_learned)

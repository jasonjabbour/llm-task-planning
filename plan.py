from worldmodel import *
from copy import deepcopy
from collections import deque

def convert_state_to_hashable(state):
    """
    Convert state dictionary to a hashable representation.
    """
    hashable_state = frozenset(state.items())
    return hashable_state

def bfs_plan(state0, actions_list, goal_fluent, max_iters=50000000):
    """
    Perform BFS to find a sequence of actions that satisfies the goal state.

    Args:
        state0 (dict): Initial state as a dictionary of fluents.
        actions_list (list): List of possible actions.
        goal_fluent (list): List of goal fluents to check against.
        max_iters (int): Maximum number of iterations before terminating.
        debug_callback (callable): Optional callback for debugging when exceptions occur.

    Returns:
        list: Sequence of actions to reach the goal.
        dict: Final state after reaching the goal.
    """
    # BFS setup
    start = ()
    states = {start: deepcopy(state0)}
    queue = deque([start])
    visited = set()
    search_iters = 0
    # print()
    while queue:
        search_iters += 1
        if search_iters > max_iters:
            print("Max iterations reached.")
            # breakpoint()
            return ["no-op"], states[start]

        node = queue.popleft()
        current_state = states[node]
        current_state_hashable = convert_state_to_hashable(current_state)

        if current_state_hashable in visited:
            continue

        visited.add(current_state_hashable)
        
        # Check if the current state satisfies the goal
        if check_goal(current_state, goal_fluent):
            # breakpoint()
            print("Goal reached!")
            breakpoint()
            return list(node), current_state

        action_num = 0
        # breakpoint()
        # Expand the current node
        for action in actions_list:
            print(action)
            action_num += 1
            print(action_num)
            new_state = transition_model(deepcopy(current_state), action)
            # breakpoint()

            if new_state and (new_state != current_state):
                # breakpoint()
                new_node = node + (action,)
                states[new_node] = deepcopy(new_state)

                queue.append(new_node)

                # Check if this new state satisfies the goal
                if check_goal(new_state, goal_fluent):
                    print("Goal reached!")
                    breakpoint()
                    return list(new_node), new_state

    print("Goal not found within the iteration limit.")
    return ["no-op"], states[start]

# Example usage
initial_state = {
    "at(P, attic: r)": True,
    "at(chest: c, attic: r)": True,
    "at(TextWorld style chest: c, attic: r)": True,
    "at(workbench: s, attic: r)": True,
    "closed(chest: c)": True,
    "in(key: k, TextWorld style chest: c)": True,
    "locked(TextWorld style chest: c)": True,
    "match(key: k, chest: c)": True,
    "match(TextWorld style key: k, TextWorld style chest: c)": True,
    "on(TextWorld style key: k, workbench: s)": True
}

# actions_list = [
#     "take TextWorld style key from workbench",
#     "unlock TextWorld style chest",
#     "open TextWorld style chest",
#     "take key from TextWorld style chest",
#     "lock the chest with the key"
# ]

actions_list = ['close chest', 'close TextWorld style chest', 'open chest', 'open TextWorld style chest', 'take key', 'take TextWorld style key', 'put key', 'put TextWorld style key', 'lock chest', 'lock TextWorld style chest', 'unlock chest', 'unlock TextWorld style chest', 'go north', 'go south', 'go east', 'go west', 'examine chest', 'examine TextWorld style chest', 'examine key', 'examine TextWorld style key', 'examine workbench', 'look', 'inventory', 'take key', 'take TextWorld style key', 'take key from chest', 'take key from TextWorld style chest', 'take key from workbench', 'take TextWorld style key from chest', 'take TextWorld style key from TextWorld style chest', 'take TextWorld style key from workbench', 'drop key', 'drop TextWorld style key', 'put key on workbench', 'put TextWorld style key on workbench', 'go north', 'go south', 'go east', 'go west', 'unlock chest with key', 'unlock chest with TextWorld style key', 'unlock TextWorld style chest with key', 'unlock TextWorld style chest with TextWorld style key', 'lock the chest with the key']

goal_fluent = [
    "at(P, attic: r)",
    "at(chest: c, attic: r)",
    "in(key: k, I)",
    "match(key: k, chest: c)",
    "locked(chest: c)"
]

# Plan using BFS
plan, final_state = bfs_plan(initial_state, actions_list, goal_fluent)

print("\nPlan:", plan)
print("Final State:", final_state)

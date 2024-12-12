from worldmodel_med import *
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
initial_state = {'at(P, scullery: r)': True, 'at(formless box: c, scullery: r)': True, 'at(type 1 box: c, scullery: r)': True, 'at(shirt: o, attic: r)': True, 'at(pair of pants: o, attic: r)': True, 'at(table: s, scullery: r)': True, 'closed(door: d)': True, 'east_of(attic: r, scullery: r)': True, 'free(attic: r, vault: r)': True, 'free(bedchamber: r, pantry: r)': True, 'free(pantry: r, bedchamber: r)': True, 'free(vault: r, attic: r)': True, 'in(American limited edition keycard: k, type 1 box: c)': True, 'in(type 1 keycard: k, formless box: c)': True, 'in(formless passkey: k, I)': True, 'in(broom: o, I)': True, 'in(teacup: o, I)': True, 'link(scullery: r, American limited edition gate: d, attic: r)': True, 'link(scullery: r, door: d, bedchamber: r)': True, 'link(attic: r, American limited edition gate: d, scullery: r)': True, 'link(bedchamber: r, door: d, scullery: r)': True, 'locked(formless box: c)': True, 'locked(type 1 box: c)': True, 'locked(American limited edition gate: d)': True, 'match(American limited edition keycard: k, American limited edition gate: d)': True, 'match(formless passkey: k, formless box: c)': True, 'match(type 1 keycard: k, type 1 box: c)': True, 'north_of(scullery: r, bedchamber: r)': True, 'north_of(attic: r, vault: r)': True, 'north_of(bedchamber: r, pantry: r)': True, 'south_of(vault: r, attic: r)': True, 'south_of(pantry: r, bedchamber: r)': True, 'south_of(bedchamber: r, scullery: r)': True, 'west_of(scullery: r, attic: r)': True}

# actions_list = [
#     "take TextWorld style key from workbench",
#     "unlock TextWorld style chest",
#     "open TextWorld style chest",
#     "take key from TextWorld style chest",
#     "lock the chest with the key"
# ]

actions_list = ['close door', 'close American limited edition gate', 'close type 1 box', 'close formless box', 'open door', 'open American limited edition gate', 'open type 1 box', 'open formless box', 'take shirt', 'take pair of pants', 'take teacup', 'take broom', 'take American limited edition keycard', 'take type 1 keycard', 'take formless passkey', 'put shirt', 'put pair of pants', 'put teacup', 'put broom', 'put American limited edition keycard', 'put type 1 keycard', 'put formless passkey', 'lock door', 'lock American limited edition gate', 'lock type 1 box', 'lock formless box', 'unlock door', 'unlock American limited edition gate', 'unlock type 1 box', 'unlock formless box', 'go north', 'go south', 'go east', 'go west', 'examine door', 'examine American limited edition gate', 'examine type 1 box', 'examine formless box', 'examine shirt', 'examine pair of pants', 'examine teacup', 'examine broom', 'examine American limited edition keycard', 'examine type 1 keycard', 'examine formless passkey', 'examine table', 'look', 'inventory', 'take shirt', 'take pair of pants', 'take teacup', 'take broom', 'take American limited edition keycard', 'take type 1 keycard', 'take formless passkey', 'take shirt from type 1 box', 'take shirt from formless box', 'take shirt from table', 'take pair of pants from type 1 box', 'take pair of pants from formless box', 'take pair of pants from table', 'take teacup from type 1 box', 'take teacup from formless box', 'take teacup from table', 'take broom from type 1 box', 'take broom from formless box', 'take broom from table', 'take American limited edition keycard from type 1 box', 'take American limited edition keycard from formless box', 'take American limited edition keycard from table', 'take type 1 keycard from type 1 box', 'take type 1 keycard from formless box', 'take type 1 keycard from table', 'take formless passkey from type 1 box', 'take formless passkey from formless box', 'take formless passkey from table', 'drop shirt', 'drop pair of pants', 'drop teacup', 'drop broom', 'drop American limited edition keycard', 'drop type 1 keycard', 'drop formless passkey', 'put shirt on table', 'put pair of pants on table', 'put teacup on table', 'put broom on table', 'put American limited edition keycard on table', 'put type 1 keycard on table', 'put formless passkey on table', 'go north', 'go south', 'go east', 'go west', 'unlock type 1 box with American limited edition keycard', 'unlock type 1 box with type 1 keycard', 'unlock type 1 box with formless passkey', 'unlock formless box with American limited edition keycard', 'unlock formless box with type 1 keycard', 'unlock formless box with formless passkey']

goal_fluent = ['at(P, attic: r)', 'in(shirt: o, I)']

# Plan using BFS
plan, final_state = bfs_plan(initial_state, actions_list, goal_fluent)

print("\nPlan:", plan)
print("Final State:", final_state)

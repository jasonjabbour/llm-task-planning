from worldmodel import *
# Initial State
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

# Test a Sequence of Actions
actions = [
    "take TextWorld style key from workbench",
    "unlock TextWorld style chest",
    "open TextWorld style chest",
    "take key from TextWorld style chest",
    "lock the chest with the key"
]

state = initial_state.copy()
print("ORIGINAL STATE:", state)

for action in actions:
    print(f"\nApplying action: {action}")

    # Get updated state
    state = transition_model(state, action)

    # Calculate changes
    # added = {key: value for key, value in state.items() if key not in state or state[key] != value}
    # removed = {key: value for key, value in state.items() if key not in state or state[key] != value}

    # # Print state changes
    # print("Added Fluents:", added)
    # print("Removed Fluents:", removed)

    # Update state
    print(f"Updated State Keys: {len(state.keys())}")

# Goal Checking
goal_fluent = ['at(P, attic: r)', 'at(chest: c, attic: r)', 'in(key: k, I)', 'match(key: k, chest: c)', 'locked(chest: c)']

def check_goal(state, goal_fluent):
    """
    Verify if the goal fluents are satisfied in the given state.

    Args:
        state (dict): Current state as a dictionary of fluents.
        goal_fluent (list): List of goal fluents in string format.

    Returns:
        bool: True if all goal fluents are satisfied, False otherwise.
    """
    return all(state.get(fluent, False) for fluent in goal_fluent)

# Check if the goal is reached
goal_reached = check_goal(state, goal_fluent)
print("\nGoal Reached:", goal_reached)

print(state)

breakpoint()
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

# Transition Model
def transition_model(state, action):
    new_state = state.copy()

    if action == "take TextWorld style key from workbench":
        if state.get("on(TextWorld style key: k, workbench: s)", False):
            new_state["on(TextWorld style key: k, workbench: s)"] = False
            new_state["in(key: k, I)"] = True

    elif action == "unlock TextWorld style chest":
        if state.get("in(key: k, I)", False) and state.get("locked(TextWorld style chest: c)", True):
            new_state["locked(TextWorld style chest: c)"] = False

    elif action == "open TextWorld style chest":
        if not state.get("locked(TextWorld style chest: c)", False):
            new_state["closed(chest: c)"] = False

    elif action == "take key from TextWorld style chest":
        if state.get("in(key: k, TextWorld style chest: c)", False) and not state.get("closed(chest: c)", True):
            new_state["in(key: k, TextWorld style chest: c)"] = False
            new_state["in(key: k, I)"] = True

    elif action == "lock the chest with the key":
        if state.get("in(key: k, I)", False) and not state.get("closed(chest: c)", True):
            new_state["locked(chest: c)"] = True

    return new_state

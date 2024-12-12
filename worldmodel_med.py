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
    # Initialize the new state with the same fluents as the current state
    new_state = state.copy()

    # Process the action
    if action == 'close door':
        new_state['closed(door: d)'] = True
    elif action == 'close American limited edition gate':
        new_state['locked(American limited edition gate: d)'] = True
    elif action == 'close type 1 box':
        new_state['locked(type 1 box: c)'] = True
    elif action == 'close formless box':
        new_state['locked(formless box: c)'] = True
    elif action == 'open door':
        new_state['closed(door: d)'] = False
    elif action == 'open American limited edition gate':
        new_state['locked(American limited edition gate: d)'] = False
    elif action == 'open type 1 box':
        new_state['locked(type 1 box: c)'] = False
    elif action == 'open formless box':
        new_state['locked(formless box: c)'] = False
    elif action == 'take shirt':
        new_state['at(shirt: o, I)'] = True
        new_state['in(shirt: o, I)'] = True
        new_state['at(shirt: o, scullery: r)'] = False
    elif action == 'take pair of pants':
        new_state['at(pair of pants: o, I)'] = True
        new_state['in(pair of pants: o, I)'] = True
        new_state['at(pair of pants: o, scullery: r)'] = False
    elif action == 'take teacup':
        new_state['at(teacup: o, I)'] = True
        new_state['in(teacup: o, I)'] = True
        new_state['at(teacup: o, scullery: r)'] = False
    elif action == 'take broom':
        new_state['at(broom: o, I)'] = True
        new_state['in(broom: o, I)'] = True
        new_state['at(broom: o, scullery: r)'] = False
    elif action == 'take American limited edition keycard':
        new_state['in(American limited edition keycard: k, I)'] = True
        new_state['in(American limited edition keycard: k, type 1 box: c)'] = False
    elif action == 'take type 1 keycard':
        new_state['in(type 1 keycard: k, I)'] = True
        new_state['in(type 1 keycard: k, formless box: c)'] = False
    elif action == 'take formless passkey':
        new_state['in(formless passkey: k, I)'] = True
        new_state['in(formless passkey: k, formless box: c)'] = False
    elif action == 'put shirt':
        new_state['at(shirt: o, I)'] = False
        new_state['at(shirt: o, scullery: r)'] = True
    elif action == 'put pair of pants':
        new_state['at(pair of pants: o, I)'] = False
        new_state['at(pair of pants: o, scullery: r)'] = True
    elif action == 'put teacup':
        new_state['at(teacup: o, I)'] = False
        new_state['at(teacup: o, scullery: r)'] = True
    elif action == 'put broom':
        new_state['at(broom: o, I)'] = False
        new_state['at(broom: o, scullery: r)'] = True
    elif action == 'put American limited edition keycard':
        new_state['in(American limited edition keycard: k, I)'] = False
        new_state['in(American limited edition keycard: k, type 1 box: c)'] = True
    elif action == 'put type 1 keycard':
        new_state['in(type 1 keycard: k, I)'] = False
        new_state['in(type 1 keycard: k, formless box: c)'] = True
    elif action == 'put formless passkey':
        new_state['in(formless passkey: k, I)'] = False
        new_state['in(formless passkey: k, formless box: c)'] = True
    elif action == 'go north':
        new_state['north_of(scullery: r, attic: r)'] = True
    elif action == 'go south':
        new_state['south_of(scullery: r, bedchamber: r)'] = True
    elif action == 'go east':
        new_state['east_of(attic: r, scullery: r)'] = True
    elif action == 'go west':
        new_state['west_of(scullery: r, attic: r)'] = True
    elif action == 'unlock type 1 box with American limited edition keycard':
        new_state['match(American limited edition keycard: k, American limited edition gate: d)'] = True
        new_state['unlock(type 1 box: c)'] = True
    elif action == 'unlock type 1 box with type 1 keycard':
        new_state['match(type 1 keycard: k, type 1 box: c)'] = True
        new_state['unlock(type 1 box: c)'] = True
    elif action == 'unlock type 1 box with formless passkey':
        new_state['match(formless passkey: k, formless box: c)'] = True
        new_state['unlock(type 1 box: c)'] = True
    elif action == 'unlock formless box with American limited edition keycard':
        new_state['match(American limited edition keycard: k, American limited edition gate: d)'] = True
        new_state['unlock(formless box: c)'] = True
    elif action == 'unlock formless box with type 1 keycard':
        new_state['match(type 1 keycard: k, type 1 box: c)'] = True
        new_state['unlock(formless box: c)'] = True
    elif action == 'unlock formless box with formless passkey':
        new_state['match(formless passkey: k, formless box: c)'] = True
        new_state['unlock(formless box: c)'] = True

    return new_state
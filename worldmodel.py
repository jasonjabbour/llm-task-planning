# make sure to include these import statements
from predicates import *
from copy import deepcopy
from utils import directions

def transition_model(state, action):
    """
    Applies the given action to the current state and returns the next state.
    
    Parameters:
    - state (dict): The current state of the game.
    - action (str): The action to be performed ('up', 'down', 'left', 'right').

    Returns:
    - dict: The next state after applying the action.
    """
    # Deep copy the state to avoid mutating the original state
    next_state = deepcopy(state)
    
    # Get the movement direction
    if action not in directions:
        # Invalid action; return state unchanged
        return next_state
    move = directions[action]
    
    # Helper function to move an object
    def move_object(obj_list, dx, dy):
        moved = False
        new_positions = []
        for pos in obj_list:
            new_x = pos[0] + dx
            new_y = pos[1] + dy
            # Check for collision with borders
            if [new_x, new_y] in next_state.get('border', []):
                new_positions.append(pos)  # Movement blocked
                continue
            # Check for pushable objects
            collision = False
            for key in next_state.get('pushables', []):
                if [new_x, new_y] in next_state.get(key, []):
                    # Attempt to push the object
                    pushed = push_object(key, new_x, new_y, dx, dy)
                    if not pushed:
                        # Cannot push; movement blocked
                        collision = True
                        break
            if collision:
                new_positions.append(pos)  # Movement blocked
                continue
            # Move the object
            new_positions.append([new_x, new_y])
            moved = True
        return new_positions, moved
    
    # Helper function to push an object
    def push_object(obj_key, x, y, dx, dy):
        obj_positions = next_state.get(obj_key, [])
        if [x, y] not in obj_positions:
            return False
        new_x = x + dx
        new_y = y + dy
        # Check if the new position is blocked
        if [new_x, new_y] in next_state.get('border', []):
            return False
        for key in next_state.get('pushables', []):
            if [new_x, new_y] in next_state.get(key, []):
                # Recursive push
                pushed = push_object(key, new_x, new_y, dx, dy)
                if not pushed:
                    return False
        # Push is possible; update the object's position
        obj_positions.remove([x, y])
        obj_positions.append([new_x, new_y])
        return True
    
    # Iterate over all controllable objects
    for controllable in next_state.get('controllables', []):
        obj_positions = next_state.get(controllable, [])
        updated_positions, moved = move_object(obj_positions, move[0], move[1])
        next_state[controllable] = updated_positions
    
    # Handle 'won' and 'lost' states if necessary
    if next_state.get('won', False) or next_state.get('lost', False):
        # Game has ended; no further actions
        return next_state

    # Process rule formation or transformation if needed
    # Checking if the rule "rock_word is_word flag_word" is formed correctly
    if rule_formed(next_state, "rock_word", "is_word", "flag_word"):
        # The rule is formed, change "rock_obj" into "flag_obj" in the state
        if "rock_obj" in next_state:
            next_state["flag_obj"] = next_state.pop("rock_obj")

    # Checking if the rule "rock_word is_word flag_word" is formed correctly
    if rule_formed(next_state, "rock_word", "is_word", "push_word"):
        # The rule is formed, change "rock_obj" into "flag_obj" in the state
        next_state["pushables"].append("rock_obj")
    
    # Process rules to update 'controllables' and 'lost' state
    new_controllables = []
    potential_rules = generate_potential_rules(next_state)
    
    for rule in potential_rules:
        word1, word2, word3 = rule
        # Check for "X is YOU" rules
        if word2 == 'is_word' and word3 == 'you_word' and rule_formed(next_state, word1, word2, word3):
            # Extract the base word to find the corresponding object
            base_word = word1.replace('_word', '')
            object_key = f"{base_word}_obj"
            if object_key in next_state:
                new_controllables.append(object_key)
    
    # Remove duplicates by converting to a set
    new_controllables = list(set(new_controllables))
    next_state['controllables'] = new_controllables
    
    # If there are no controllable objects, set 'lost' to True
    if not next_state['controllables']:
        next_state['lost'] = True
    else:
        next_state['lost'] = False

     # Ensure 'goop_obj' updates correctly after movement if 'baba_obj' overlaps with 'goop_obj'
    if 'baba_obj' in next_state and 'goop_obj' in next_state and rule_formed(next_state, "goop_word", "is_word", "sink_word"):
        for i, baba_pos in enumerate(next_state['baba_obj']):
            for j, goop_pos in enumerate(next_state['goop_obj']):
                if overlapping(next_state, 'baba_obj', i, 'goop_obj', j):
                    # Adjust goop_obj state appropriately based on the interaction (e.g., sinking baba)
                    next_state['lost'] = True  # Baba gets lost when it touches goop_obj

     # Ensure rock obj and goop obj are removed if they are overlapping when rule_formed goop_word is_word sink_word
    if 'rock_obj' in next_state and 'goop_obj' in next_state and rule_formed(next_state, "goop_word", "is_word", "sink_word"):
        for i, rock_pos in enumerate(next_state['rock_obj']):
            for j, goop_pos in enumerate(next_state['goop_obj']):
                if overlapping(next_state, 'rock_obj', i, 'goop_obj', j):
                    next_state['rock_obj'].remove(rock_pos)
                    next_state['goop_obj'].remove(goop_pos)
                    

    return next_state

import re

def convert_admissible_to_facts(admissible_commands):
    """
    Converts natural language admissible commands into logical fact representations.

    Args:
        admissible_commands (list): List of natural language commands.

    Returns:
        list: Logical representations of the commands.
    """
    logical_commands = []
    for command in admissible_commands:
        parts = command.split(" ")

        if parts[0] == "look":
            logical_commands.append("look()")
        elif parts[0] == "goal":
            logical_commands.append("goal()")
        elif parts[0] == "inventory":
            logical_commands.append("inventory()")
        elif parts[0] == "go":
            logical_commands.append(f"go({parts[1]})")
        elif parts[0] == "examine":
            logical_commands.append(f"examine({parts[1]})")
        elif parts[0] == "eat":
            logical_commands.append(f"eat({parts[1]})")
        elif parts[0] == "open":
            logical_commands.append(f"open({parts[1]})")
        elif parts[0] == "close":
            logical_commands.append(f"close({parts[1]})")
        elif parts[0] == "drop":
            logical_commands.append(f"drop({parts[1]})")
        elif parts[0] == "take" and "from" not in parts:
            logical_commands.append(f"take({parts[1]})")
        elif parts[0] == "put":
            logical_commands.append(f"put({parts[1]}, {parts[-1]})")
        elif parts[0] == "take" and "from" in parts:
            logical_commands.append(f"take({parts[1]}, {parts[-1]})")
        elif parts[0] == "insert":
            logical_commands.append(f"insert({parts[1]}, {parts[-1]})")
        elif parts[0] == "lock":
            logical_commands.append(f"lock({parts[1]}, {parts[-1]})")
        elif parts[0] == "unlock":
            logical_commands.append(f"unlock({parts[1]}, {parts[-1]})")
        else:
            logical_commands.append(f"action({command})")  # Fallback for unknown commands

    return logical_commands


def extract_code_from_response(response):
    """
    Extract Python code from the LLM response.

    Args:
        response (str): Response from the LLM.

    Returns:
        str: Extracted Python code or None if not found.
    """
    code_match = re.search(r'```python(.*?)```', response, re.IGNORECASE | re.DOTALL)
    if code_match:
        return code_match.group(1).strip()
    else:
        return None

def overwrite_world_model(new_code, model_name, env_id):
    """
    Save the generated world model to a Python file.

    Args:
        new_code (str): Python code to save.
        model_name (str): Name of the model used.
        env_id (str): ID of the environment.
    """
    world_model_path = f"worldmodel_{model_name}_{env_id}.py"
    with open(world_model_path, 'w') as file:
        file.write(new_code)
    print(f"World model saved to {world_model_path}.")
import os
import textworld.gym
import re
from groq import Groq
from utils import *

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# List of models and environments
models = ["llama3-8b-8192"]

environments = [
    {"id": "env_easy", "difficulty": 1, "game_path": "tw_games/easy.z8"},
    # {"id": "env_medium", "difficulty": 2, "game_path": "tw_games/medium.z8"},
    # {"id": "env_hard", "difficulty": 3, "game_path": "tw_games/hard.z8"}
]

# Configure EnvInfos with desired parameters
request_infos = textworld.EnvInfos(
    description=True,
    inventory=True,
    entities=True,
    admissible_commands=True,
    facts=True,
    objective=True,
    verbs=True,
    win_facts=True,
    fail_facts=True
)

# Hardcoded entity types for environments
ENTITY_TYPES = {
    "env_easy": {
        # Containers
        "chest": "c",
        "TextWorld style chest": "c",

        # Keys
        "key": "k",
        "TextWorld style key": "k",

        # Supporters
        "workbench": "s",

        # Directions
        "north": "dir",
        "south": "dir",
        "east": "dir",
        "west": "dir"
    },
    "env_medium": {
        # Doors and gateways
        "door": "d",
        "American limited edition gate": "d",

        # Containers
        "type 1 box": "c",
        "formless box": "c",

        # Objects
        "shirt": "o",
        "pair of pants": "o",
        "teacup": "o",
        "broom": "o",

        # Keys
        "American limited edition keycard": "k",
        "type 1 keycard": "k",
        "formless passkey": "k",

        # Supporters
        "table": "s",

        # Directions
        "north": "dir",
        "south": "dir",
        "east": "dir",
        "west": "dir"
    },
    "env_hard": {
        # Doors and gateways
        "hatch": "d",
        "gate": "d",
        "passageway": "d",
        "portal": "d",
        "door": "d",
        "gateway": "d",
        "stone gateway": "d",

        # Containers
        "safe": "c",
        "American limited edition chest": "c",
        "Canadian style box": "c",
        "suitcase": "c",
        "box": "c",
        "cabinet": "c",
        "locker": "c",
        "American style box": "c",
        "fudge scented safe": "c",
        "American limited edition locker": "c",
        "type Y chest": "c",
        "American locker": "c",
        "basket": "c",
        "type O chest": "c",

        # Keys
        "type O latchkey": "k",
        "passkey": "k",
        "American latchkey": "k",
        "latchkey": "k",
        "American limited edition keycard": "k",
        "Canadian style keycard": "k",
        "keycard": "k",
        "American style latchkey": "k",
        "fudge scented passkey": "k",
        "American limited edition key": "k",
        "type Y key": "k",

        # Objects
        "paper towel": "o",
        "teapot": "o",

        # Supporters
        "workbench": "s",
        "table": "s",

        # Food
        "loaf of bread": "f",

        # Directions
        "north": "dir",
        "south": "dir",
        "east": "dir",
        "west": "dir"
    }
}

def generate_pruned_actions(verbs, entities, entity_types, verb_constraints, facts_dict):
    actions = []

    # Handle single-argument actions
    for verb, allowed_types in verb_constraints.items():
        if not allowed_types:
            # Verbs like "look", "inventory" that don't require entities
            actions.append(verb)
        else:
            for entity, entity_type in entity_types.items():
                if entity_type in allowed_types:
                    actions.append(f"{verb} {entity}")

    # Handle "take X" for directly accessible items
    for obj, obj_type in entity_types.items():
        if obj_type in ["o", "k"]:  # Objects or keys
            actions.append(f"take {obj}")

    # Handle "take X from Y"
    for obj, obj_type in entity_types.items():
        if obj_type in ["o", "k"]:  # Objects or keys
            for target, target_type in entity_types.items():
                if target_type in ["c", "s"]:  # Containers or supporters
                    actions.append(f"take {obj} from {target}")

    # Handle "drop X" for inventory items
    for obj, obj_type in entity_types.items():
        if obj_type in ["o", "k"]:
            actions.append(f"drop {obj}")

    # Handle "put X on Y"
    for obj, obj_type in entity_types.items():
        if obj_type in ["o", "k"]:  # Objects or keys
            for supporter, supporter_type in entity_types.items():
                if supporter_type == "s":  # Supporters only
                    actions.append(f"put {obj} on {supporter}")

    # Handle "go X" for directions
    for entity, entity_type in entity_types.items():
        if entity_type == "dir":
            actions.append(f"go {entity}")

    # Handle "unlock X with Y"
    for container, container_type in entity_types.items():
        if container_type == "c":  # Containers
            for key, key_type in entity_types.items():
                if key_type == "k":  # Keys
                    actions.append(f"unlock {container} with {key}")

    return actions



# Run the environment
for model in models:
    print(f' --- Generating World Model with Model: {model} ---')
    for env in environments:
        env_id = env["id"]
        print(f' --- Environment: {env} ---')
        # Register the game and create the environment
        game_env_id = textworld.gym.register_game(env["game_path"], request_infos, max_episode_steps=20)
        textworld_env = textworld.gym.make(game_env_id)

        print(f'---- Initializing World Model ----')
        obs, infos = textworld_env.reset()
        textworld_env.render()

        # Convert facts to a dictionary
        facts_dict = {str(fact): True for fact in infos["facts"]}

        # Use hardcoded entity types based on environment
        entity_types = ENTITY_TYPES[env_id]

        # Define verb constraints
        verb_constraints = {
            "close": ["c", "d"],  # Containers and doors
            "open": ["c", "d"],   # Containers and doors
            "take": ["o", "k"],   # Objects and keys
            "put": ["o", "k"],    # Objects and keys
            "lock": ["c", "d"],   # Containers and doors
            "unlock": ["c", "d"], # Containers and doors
            "go": ["dir"],        # Only directions
            "examine": ["o", "c", "s", "k", "d"],  # Objects, containers, supporters, keys, and doors
            "look": [],           # No specific target
            "inventory": []       # No specific target
        }

        # Generate pruned actions
        pruned_actions = generate_pruned_actions(infos["verbs"], infos["entities"], entity_types, verb_constraints, facts_dict)

        # Compare pruned actions with admissible commands
        missing_actions = [cmd for cmd in infos["admissible_commands"] if cmd not in pruned_actions]
        if missing_actions:
            print("Missing actions:", missing_actions)
        else:
            print("All admissible commands are captured!")

        # Prepare the prompt for the LLM
        goal_fluent = [str(fact) for condition in infos["win_facts"] for nested_facts in condition for fact in nested_facts]
        entities_list = ', '.join(infos["entities"])
        inventory_list = infos['inventory']

        prompt_content = (
                f"Build a Python function that serves as a transition model for this environment. \n"
                f"The transition function must explicitly handle every action in the Actions list, regardless of its perceived relevance to the goal. Even if an action seems redundant or unrelated to the goal, include a corresponding if statement in the transition model. If an action does not affect the state, return the state unchanged for that action. \n"
                f"Here is some information for your goal, state, etc., in this environment. "
                f"You will need to use the symbolic version in your transition function.\n\n"
                f"Goal (in natural language): {infos['objective']}\n\n"
                f"Goal (symbolic):\n{goal_fluent}\n\n"
                f"State (in natural language): {obs}\n\n"
                f"State (symbolic):\n{facts_dict}\n\n"
                f"Actions list (Your transition model MUST HANDLE ALL THESE ACTIONS, Even actions perceived as synonyms or unnecessary must still be handled!):\n{pruned_actions}\n\n"
                f"Entities in the room: {entities_list}\n\n"
                f"Inventory: {inventory_list}\n\n"
                f"Please write the Python code for the transition model below. Include the following utility function for verifying the goal state. You do not need to model rewards:\n\n"
                f"RESPONSE FORMAT:\n"
                f"```python\n"
                f"def check_goal(state, goal_fluent):\n"
                f"    \"\"\"\n"
                f"    Verify if the goal fluents are satisfied in the given state.\n\n"
                f"    Args:\n"
                f"        state (dict): Current state as a dictionary of fluents.\n"
                f"        goal_fluent (list): List of goal fluents in string format.\n\n"
                f"    Returns:\n"
                f"        bool: True if all goal fluents are satisfied, False otherwise.\n"
                f"    \"\"\"\n"
                f"    return all(state.get(fluent, False) for fluent in goal_fluent)\n\n"
                f"# Transition Model\n"
                f"def transition_model(state, action):\n"
                f"    # Your generated transition model logic goes here.\n"
                f"    pass\n"
                f"```"
            )


        print("PROMPT BELOW:")
        print(prompt_content)

        breakpoint()

        # Generate the world model
        messages = [{"role": "user", "content": prompt_content}]
        chat_completion = client.chat.completions.create(
            messages=messages,
            model=model
        )
        response = chat_completion.choices[0].message.content.strip()

        print("Generated Response:", response)

        # Extract the Python code from the response
        world_model_code = extract_code_from_response(response)

        if world_model_code:
            overwrite_world_model(world_model_code, model_name=model, env_id=env_id)
        else:
            print("No valid Python code found in the response.")

        # Stop after generating the world model
        break

print("World model generation completed.")

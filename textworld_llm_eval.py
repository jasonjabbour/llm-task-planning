import os
import csv
import textworld.gym
import time
from groq import Groq

# Initialize the Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

# List of models and environments
# models = ["mixtral-8x7b-32768", "llama3-8b-8192", "gemma-7b-it"]

models = ["mixtral-8x7b-32768"]

environments = [
    # {"id": "env_easy", "difficulty": 1, "game_path": "tw_games/easy.z8"},
    # {"id": "env_medium", "difficulty": 2, "game_path": "tw_games/medium.z8"},
    {"id": "env_hard", "difficulty": 3, "game_path": "tw_games/hard.z8"},
]

# Request additional information for TextWorld environments
request_infos = textworld.EnvInfos(
    admissible_commands=True,
    entities=True
)

# CSV file to store results
csv_file = "experiment_results.csv"

# Check if the CSV file exists and prepare it accordingly
file_exists = os.path.exists(csv_file)

with open(csv_file, mode="a", newline="") as file:  # Open file in append mode
    writer = csv.writer(file)
    if not file_exists:
        # Write header row if file does not exist
        writer.writerow(["Model", "Environment", "Difficulty", "Run", "Moves", "Score"])

# Run experiments
for model in models:
    print(f' --- Experiments with Model: {model} ---')
    for env in environments:
        print(f' --- Environment: {env} ---')
        # Register the game and create the environment
        env_id = textworld.gym.register_game(env["game_path"], request_infos, max_episode_steps=40)
        textworld_env = textworld.gym.make(env_id)

        for run in range(1, 2):  # Run 5 trials per model per environment
            print(f'---- Trial: {run} ----')
            obs, infos = textworld_env.reset()
            textworld_env.render()

            score, moves, done = 0, 0, False
            messages = []  # Store interaction history for the model

            while not done:
                # Prepare the prompt for the LLM by including entities and admissible commands
                entities_list = ', '.join(infos["entities"])
                commands_list = '\n  '.join(infos["admissible_commands"])

                prompt_content = f"Observation: {obs}\nEntities in the room: {entities_list}\nAdmissible commands:\n  {commands_list}\n\nPlease select one of the admissible commands listed above as your next action, and output only the chosen command:"

                # Use the model to generate the next action
                if moves == 0:
                    messages = [{"role": "user", "content": prompt_content}]
                else:
                    messages.append({"role": "user", "content": prompt_content})


                # Check message length and manage context size
                if len(messages) > 20:
                    # Keep only the most recent 20 messages
                    messages = messages[-20:]

                # Call the language model for the next action
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=model
                )
                action = chat_completion.choices[0].message.content.strip()

                messages.append({"role": "assistant", "content": action})

                # Execute the action in the environment
                obs, score, done, infos = textworld_env.step(action)
                textworld_env.render()
                moves += 1

                # GROQ Limits to 30 requests per minute
                time.sleep(1.1)

            # Log results
            with open(csv_file, mode="a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow([model, env["id"], env["difficulty"], run, moves, score])

        textworld_env.close()

print(f"Experiment results saved to {csv_file}.")
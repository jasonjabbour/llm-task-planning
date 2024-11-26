import textworld.gym


request_infos = textworld.EnvInfos(
    admissible_commands=True,  # All commands relevant to the current state.
    entities=True              # List of all interactable entities found in the game.
)

# Requesting additional information should be done when registering the game.
env_id = textworld.gym.register_game('tw_games/custom_game.z8', request_infos,  max_episode_steps=50)
# Start the environment.
env = textworld.gym.make(env_id)

obs, infos = env.reset()


env.render()

score, moves, done = 0, 0, False
while not done:
    print("Entities: {}\n".format(infos["entities"]))
    print("Admissible commands:\n  {}".format("\n  ".join(infos["admissible_commands"])))
    # command = input(">")
    command = str(moves)
    obs, score, done, infos = env.step(command)
    env.render()
    moves += 1

env.close()
print("moves: {}; score: {}".format(moves, score))
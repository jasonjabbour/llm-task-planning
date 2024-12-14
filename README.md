# llm-task-planning
LLM Task Planning

# Install

sudo apt update && sudo apt install build-essential libffi-dev python3-dev curl git

conda create --name llm-tp python=3.10

pip install textworld

pip install 'transformers[torch]'

pip install tiktoken

pip install blobfile

pip install sentencepiece

pip install optimum

pip install auto-gptq

# Running using Groq
pip install groq

export GROQ_API_KEY=<groq_token_here>


# Generate an Environment

tw-make custom --world-size 5 --nb-objects 10 --quest-length 10 --seed 1234 --output tw_games/custom_game.z8

# Installing Symbolic Planner (Fast Downward)

Follow the link below to install the Fast Downward planner that will allow you to run high_level_planner.sh to output the plan_demo_LEVELS.json file.

https://github.com/aibasel/downward.git

# run the prompts and copy paste the transition model into world model

# evaluating transition model (example for level 0)

python prism.py --levels "[('demo_LEVELS', 0)]"

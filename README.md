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


# Installing Symbolic Planner (Fast Downward)

Follow the link below to install the Fast Downward planner that will allow you to run high_level_planner.sh to output the plan_demo_LEVELS.json file and the exploration_generation.sh.

https://github.com/aibasel/downward.git

# Evaluating transition model (example for level 0)

python prism.py --levels "[('demo_LEVELS', 0)]" 

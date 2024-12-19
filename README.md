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

# run the prompts and copy paste the transition model into world model

# evaluating transition model (example for level 0)

python prism.py --levels "[('demo_LEVELS', 0)]" 

- After generating the transition models with the prompts you can use this to evaluate the corresponding levels by changing the level parameter.
- You may get to a level where the transition model no longer works, in which case you will need to refine it with the prompts

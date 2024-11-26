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



# Generate an Environment
tw-make custom --world-size 5 --nb-objects 10 --quest-length 10 --seed 1234 --output tw_games/custom_game.z8
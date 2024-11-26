import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Set the model directory
model_dir = "./local_llms/models--TheBloke--Llama-2-7B-GGML"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model = AutoModelForCausalLM.from_pretrained(model_dir)

# Ensure the model is on GPU
device = torch.device("cuda")
model.to(device)

# Function to encode input and generate model response
def chat_with_model(prompt):
    inputs = tokenizer.encode(prompt, return_tensors="pt").to(device)
    outputs = model.generate(inputs, max_length=512)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example interaction
prompt = "Hello, how are you?"
response = chat_with_model(prompt)
print(response)


# import os
# import torch
# from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# torch.cuda.empty_cache()  # Attempt to free up unused memory
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'  # Memory optimization

# if torch.cuda.is_available():
#     print(torch.cuda.memory_summary())  # Detailed memory usage


# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print(f"Using device: {device}")

# base_dir = "./local_llms"

# model_options = [
#     {"name": "EleutherAI/gpt-j-6B", "path": os.path.join(base_dir, "gpt-j-6B")},
#     {"name": "yahma/llama-13b-hf", "path": os.path.join(base_dir, "llama-13b-hf")},
#     {"name": "google/gemma-7b", "path": os.path.join(base_dir, "gemma-7b")},
#     {"name": "Llama-7B-2Quant", "path": os.path.join(base_dir, "models--TheBloke--Llama-2-7B-GGML")},
# ]

# for model in model_options:
#     config_path = os.path.join(model["path"], "config.json")
#     if os.path.exists(config_path):
#         model["config"] = AutoConfig.from_pretrained(config_path)
#     else:
#         print(f"Warning: No config.json found in {model['path']}")

# print("Available models:")
# for idx, model in enumerate(model_options):
#     print(f"{idx + 1}: {model['name']}")

# choice = int(input("Select a model by number: ")) - 1
# selected_model = model_options[choice]

# try:
#     if "config" in selected_model:
#         model = AutoModelForCausalLM.from_config(selected_model["config"]).to(device)
#     else:
#         model = AutoModelForCausalLM.from_pretrained(selected_model["path"]).to(device)
#     tokenizer = AutoTokenizer.from_pretrained(selected_model["path"])
# except Exception as e:
#     print(f"Failed to load model or tokenizer: {e}")
#     exit(1)

# print("\nChat loop started. Type 'exit' to end.")
# while True:
#     user_input = input("You: ")
#     if user_input.lower() == "exit":
#         print("Exiting chat.")
#         break

#     input_ids = tokenizer.encode(user_input, return_tensors="pt").to(device)
#     outputs = model.generate(input_ids, max_length=100, num_return_sequences=1)
#     response = tokenizer.decode(outputs[0], skip_special_tokens=True)
#     print(f"Model: {response}")
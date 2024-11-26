import os
from huggingface_hub import hf_hub_download, HfApi, HfFolder

# Authenticate with your Hugging Face token
api = HfApi()
HfFolder.save_token('your_token_here')

# Define the repository and filename
repo_id = "TheBloke/Llama-2-7B-GGML"
filename = "llama-2-7b.ggmlv3.q2_K.bin"

# Set the download directory
download_dir = "./local_llms"
os.makedirs(download_dir, exist_ok=True)  # Ensure directory exists

# Download the file
file_path = hf_hub_download(repo_id=repo_id, filename=filename, cache_dir=download_dir)
print(f"Download complete: {file_path}")

# import os
# from transformers import AutoModelForCausalLM, AutoTokenizer
# from huggingface_hub import login

# # Uncomment and use your Hugging Face token to authenticate
# login(token='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')

# # List of models to download
# model_list = [
#     "EleutherAI/gpt-j-6B",  
#     "yahma/llama-13b-hf", 
#     "google/gemma-7b"
# ]

# # Base directory for saving models
# base_dir = "./local_llms"

# # Ensure the base directory exists
# os.makedirs(base_dir, exist_ok=True)

# # Iterate through the list of models
# for model_name in model_list:
#     print(f"\n\n -- Attempting Download: {model_name} -- \n\n")
#     # Extract the directory name from the model name
#     dir_name = model_name.split("/")[-1]  # Get the last part, e.g., 'gpt-j-6B'
#     download_dir = os.path.join(base_dir, dir_name)

#     # Ensure the directory exists
#     os.makedirs(download_dir, exist_ok=True)

#     # Check if the model already exists to skip redundant downloads
#     if not os.listdir(download_dir):  # Check if directory is empty
#         print(f"Downloading {model_name} to {download_dir}...")
#         tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=download_dir)
#         model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=download_dir)
#         print(f"Download complete for {model_name}.")
#     else:
#         print(f"{model_name} already exists in {download_dir}. Skipping download.")
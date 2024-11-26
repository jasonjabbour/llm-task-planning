import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a16"
download_directory = "./local_llms"

# Ensure CUDA is available and set the default device to GPU if possible
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=download_directory)
# Set pad token to EOS token (common practice)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=download_directory,
    torch_dtype="auto",
    device_map="auto"  # This should map to the GPU if available
)
model.to(device)  # Explicitly move the model to the GPU if CUDA is available

def chat_with_model(prompt):
    # Encode input and generate attention mask
    encoded_input = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
    input_ids = encoded_input['input_ids'].to(device)
    
    # Check if pad_token_id is None and set it if necessary
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id  # Only do this if absolutely necessary
        
    # Create attention mask that ignores pad tokens
    attention_mask = encoded_input['attention_mask'].to(device)

    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_length=512,
        no_repeat_ngram_size=2,
        do_sample=True,
        top_p=0.9,
        temperature=0.6,
        pad_token_id=tokenizer.pad_token_id
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Starting a new conversation with a context setup
def start_conversation():

    conversation_history = ''

    while True:
        print("You: ", end="")
        user_input = input()

        # Update the conversation history with the user input
        conversation_history += "\n" + user_input
        # Generate model's response
        response = chat_with_model(conversation_history)
        print("Bot:", response)

        # Update conversation history with the bot's response
        conversation_history += "\n" + response

# Initialize the conversation
start_conversation()
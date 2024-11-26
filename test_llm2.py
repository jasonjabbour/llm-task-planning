from transformers import AutoTokenizer, AutoModelForCausalLM

# Define the model ID and the directory for downloading model and tokenizer
model_id = "neuralmagic/Meta-Llama-3-8B-Instruct-quantized.w8a16"
download_directory = "./model_downloads"  # Specify your desired download directory

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_id, cache_dir=download_directory)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=download_directory,
    torch_dtype="auto",  # Ensures compatibility with quantized weights
    device_map="auto"    # Maps the model to the appropriate device (CPU/GPU)
)

# Define a function to chat with the model
def chat_with_model(messages):
    # Extracting text content from messages
    text_inputs = [message['content'] for message in messages if 'content' in message]
    
    # Encode the messages using the tokenizer
    input_ids = tokenizer(text_inputs, return_tensors="pt", padding=True).input_ids.to(model.device)

    # Generate a response from the model
    outputs = model.generate(
        input_ids,
        max_new_tokens=256,  # Limits the length of the generated text
        no_repeat_ngram_size=2,
        do_sample=True,      # Enable sampling for more diverse responses
        top_p=0.9,           # Nucleus sampling
        temperature=0.6      # Controls the randomness of the output
    )

    # Decode and return the response
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Example conversation setup
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"}
]

# Chat with the model
response = chat_with_model(messages)
print("Model response:", response)

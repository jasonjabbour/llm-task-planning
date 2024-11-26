import os

from groq import Groq

# Set up the client with your API key
client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)

# Start the chat session
def chat_with_model():
    messages = []
    while True:
        # Get user input
        user_input = input("You: ")
        if user_input.lower() == "exit":
            print("Exiting chat...")
            break
        
        # Append the user's message to the conversation history
        messages.append({
            "role": "user",
            "content": user_input,
        })
        
        # Get model completion
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama3-8b-8192",
        )
        
        # Print model's response
        model_response = chat_completion.choices[0].message.content
        print("Model: ", model_response)
        
        # Append model's response to the conversation history
        messages.append({
            "role": "system",
            "content": model_response,
        })

# Call the function to start chatting
chat_with_model()
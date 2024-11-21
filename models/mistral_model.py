import requests
import os
import json

# API endpoint
api_url = "http://localhost:1234/v1/completions"

# API key if required
api_key = os.getenv("OPENAI_API_KEY", "not-needed")

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {api_key}"
}

messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "Hello, how are you?"},
]

def send_message():
    # Construct the prompt by concatenating previous messages
    prompt = ""
    for msg in messages:
        role = msg['role'].capitalize()
        content = msg['content']
        prompt += f"{role}: {content}\n"
    
    # Prepare the payload
    payload = {
        "model": "mistral-7b-instruct-v0.3:2",
        "prompt": prompt,
        "max_tokens": 100,
        "n": 1,
        "stop": None,
        "temperature": 0.3,
    }
    
    try:
        # Make the POST request to the API
        response = requests.post(api_url, headers=headers, json=payload)
        
        # Raise an exception for HTTP error codes
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        
        # Debugging: Print the entire response
        print("Full API Response:", json.dumps(data, indent=2))
        
        # Extract the assistant's response
        if 'choices' in data and len(data['choices']) > 0:
            assistant_message = data['choices'][0]['text'].strip()
            return assistant_message
        else:
            print("No 'choices' found in the response.")
            print("Response Content:", data)
            return "Sorry, I couldn't process that."
    
    except requests.exceptions.HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
        print(f"Response Content: {response.text}")
        return "Sorry, I couldn't process that."
    except Exception as err:
        print(f"An error occurred: {err}")
        return "Sorry, I couldn't process that."

# Send the message and get the response
message = send_message()

# Append the assistant's response to the messages
messages.append({"role": "assistant", "content": message})

# Print the conversation
for msg in messages:
    print(f"{msg['role'].capitalize()}: {msg['content']}")

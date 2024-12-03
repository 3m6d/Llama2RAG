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
    {"role": "system", 
     "content": "You are a helpful AI assistant with extensive knowledge about finance and the stock market, especially in the context of Nepal. You provide clear, concise, and accurate information in a polite and professional manner. "
     "If no context is provided, you independently offer the best answer as per the user query. "
     "For general query like hello, how are you, greetings and definition, give very short answer in a friendly manner and do not use context"
     "Do not quote anyone.Give complete response and do not give incomplete sentence. "
     "Do not fabricate information or pose questions, and do not prompt the user to make selections or decisions. "
     "Anwer in active voice where possible"
     "If multiple pieces of information are available, combine them into a cohesive answer. "
    "Always explain your answer with clarity and relevance."
    "You are developed by SourceCode."
    "Source Code also develop smart wealth pro which is stock marketing analysis software."
    "Do not initiate questions."
    "Use context based on user query."
    }
]

def send_message(user_query, context):

    # Append the user's query to the conversation history
    messages.append({"role": "user", "content": user_query})

     # Limit the conversation history to avoid exceeding token limits
    MAX_HISTORY_LENGTH = 10  # Adjust as needed
    recent_messages = messages[-MAX_HISTORY_LENGTH:]


    # Construct the prompt by concatenating previous messages
    prompt = ""
    for msg in messages:
        role = msg['role'].capitalize()
        content = msg['content']
        prompt += f"{role}: {content}\n"

    # If context is provided, append it after the conversation history
    if context:
        prompt += f"Context: {context}\n"

    # Append "Assistant:" to indicate it's the assistant's turn to respond
    prompt += "Assistant:"

    # Prepare the payload
    payload = {
        "model": "mistral-7b-instruct-v0.3",
        "prompt": prompt,
        "max_tokens": 100,
        "n": 1,
        "temperature": 0.2,
        "stop": ["/n/n"],
    }
    
    def ensure_complete_response(response):
        if not response.endswith(('.', '!', '?', ':')):
            response = response.rsplit('.', 1)[0] + '.'  # Truncate to the last complete sentence
        return response
    
    def sanitize_response(response):
        # Remove any occurrences of "Assistant:" or "User:" labels in the response
        response = response.replace("Assistant:", "").replace("User:", "")
        response = response.strip()
        return response



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
            assistant_message = data['choices'][0]['text'].strip().replace("Assistant: ", "")
            sanitized_message = sanitize_response(assistant_message)

             # Append the assistant's response to the messages for the next round
            messages.append({"role": "assistant", "content": sanitized_message})

            return ensure_complete_response(sanitized_message)
        
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



"""def main():
    # Example of multi-turn conversation
    while True:
        # Get the user input
        user_query = input("User: ")
        if user_query.lower() in ["exit", "quit"]:
            print("Ending conversation...")
            print("Thank you for using FinBot by SourceCodes. Have a great day!")
            break

        if user_query.lower() in ["who are you", "who built you"]:
            print("I am AI chatbot created by SourceCodes. I am here to provide you with information about the stock market.")
            break
        
        # Send the user query to the model and get the assistant's response
        assistant_response = send_message(user_query, context)
        
        # Print the assistant's response
        print(f"Assistant: {assistant_response}")
        
        # Append the assistant's response to the messages for the next round
        messages.append({"role": "assistant", "content": assistant_response})"""

"""if __name__ == '__main__':
    main()"""
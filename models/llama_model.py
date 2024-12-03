from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# Force device to be CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# Ensure the pad_token_id is set
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load the model with CPU-compatible data type
model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-v0.1",
    torch_dtype=torch.float32  # Use float32 for CPU
)

# Move the model to the CPU
model.to(device)
model.eval()

def generate_response(query, context):
    # System prompt
    system_prompt = "You are a helpful AI assistant that has base knowledge about financial worlds. You are polite. If there is no context, you independtly answer the best answer you know."

    # Construct the prompt
    if context:
        prompt = f"{system_prompt}\n\n{context}\n\nUser: {query}\nAssistant:"
    else:
        prompt = f"{system_prompt}\n\nUser: {query}\nAssistant:"

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt")

    # Move inputs to the CPU
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Generate the response
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    # Decode the generated tokens
    response = tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the assistant's reply
    assistant_reply = response.split("Assistant:")[-1].strip()

    return assistant_reply
"""

"""if __name__ == "__main__":
    user_query = "Hello, how are you?"
    context = "I am good."  # Since you've removed the get_relevant_context function
    response = generate_response(user_query, context)
    print("Assistant:", response)





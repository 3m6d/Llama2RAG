<<<<<<< HEAD
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


"""if __name__ == "__main__":
    user_query = "Hello, how are you?"
    context = "I am good."  # Since you've removed the get_relevant_context function
    response = generate_response(user_query, context)
    print("Assistant:", response)"""




=======
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# Load the tokenizer and model from Hugging Face
model_name = "meta-llama/Llama-2-7b-chat-hf"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    use_auth_token=True,
    legacy=False  # To use the new tokenizer behavior
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    use_auth_token=True,
    torch_dtype=torch.float16,  # Use float16 for efficiency
    device_map="auto"  # Automatically choose the device
)

# Set pad_token_id to eos_token_id to avoid generation issues
tokenizer.pad_token_id = tokenizer.eos_token_id

def generate_response(query, context):
    # System prompt for the assistant
    system_prompt = """As a helpful and respectful AI assistant, provide accurate and relevant information based solely on the provided document.
Responses must adhere to these guidelines:
- Be concise, factual, and limited to 2-3 sentences, up to 50 words.
- Maintain an ethical, unbiased, and positive tone; avoid harmful, offensive, or speculative content.
- Do not include introductory or confirmatory phrases like "yes" or "you are correct."
- If no relevant information is available in the document, state: "I cannot provide an answer based on the provided document."
- Do not fabricate information or pose questions, and do not prompt the user to make selections or decisions."""

    # Construct the prompt with context and question
    prompt = f"""[INST] <<SYS>>
{system_prompt}
<</SYS>>

Document:
{context}

Question:
{query}
[/INST]"""

    # Tokenize and encode the input
    inputs = tokenizer(prompt, 
                       return_tensors="pt", 
                       padding=True, 
                       truncation=True,
                       max_length=4096)
    
    # Generate the response
    output_tokens = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id
    )
    
    # Extract the assistant's response
    generated_tokens = output_tokens[0][inputs.input_ids.shape[-1]:]
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response.strip()

def get_relevant_context(user_query, document_chunks, max_chunks=3):
    # For simplicity, return the top 'max_chunks' relevant chunks
    # You can improve this by selecting chunks based on similarity scores
    return "\n".join(document_chunks[:max_chunks])
>>>>>>> 28bcd8ae440328b0690a4c2be9311d4d3004eae0

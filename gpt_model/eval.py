#%%
import torch
import tiktoken
from gpt_model.model import GPTModel
import os




# %% Load the model using config

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12, 
    "drop_rate": 0.1,
    "qkv_bias": False
}

# %% Utility functions for text to token ID conversion

def text_to_token_ids(text, tokenizer):
    """
    This function converts a text string into a tensor of token IDs using a 
    given tokenizer.

    Args:
        text (str): The input text string.
        tokenizer: The tokenizer object used to convert text to tokens.

    Returns:
        torch.Tensor: A tensor of token IDs representing the input text.
                      Shape: (1, num_tokens)
    """
    # Encode the text using the tokenizer
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})  
    # Convert the encoded list to a PyTorch tensor
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # Add a batch dimension
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    """
    This function converts a tensor of token IDs back into a text string 
    using a given tokenizer.

    Args:
        token_ids (torch.Tensor): The tensor of token IDs.
                                  Shape: (1, num_tokens)
        tokenizer: The tokenizer object used to convert tokens to text.

    Returns:
        str: The decoded text string.
    """
    # Remove the batch dimension
    flat = token_ids.squeeze(0)  
    # Decode the token IDs using the tokenizer
    return tokenizer.decode(flat.tolist())  # Convert to list for decoding

#%%

#%% Generate function

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    """
    Generates text using a language model with options for temperature-based 
    sampling and top-k filtering.

    Args:
        model: The language model to use for generating text.
        idx (torch.Tensor): The initial sequence of token indices. 
                            Shape: (batch_size, initial_sequence_length)
        max_new_tokens (int): The maximum number of tokens to generate.
        context_size (int): The size of the context window used by the model.
        temperature (float, optional): The temperature used for sampling. 
                                       Higher values increase the probability 
                                       of selecting less likely tokens. 
                                       Defaults to 0.0 (greedy decoding).
        top_k (int, optional): The number of top tokens to consider when sampling. 
                               Defaults to None (consider all tokens).
        eos_id (int, optional): The ID of the end-of-sequence token. If provided, 
                                generation stops when this token is generated. 
                                Defaults to None.

    Returns:
        torch.Tensor: The generated sequence of token indices. 
                      Shape: (batch_size, initial_sequence_length + generated_sequence_length)
    """
    for _ in range(max_new_tokens):
        # Take the last `context_size` tokens as the context
        idx_cond = idx[:, -context_size:]  
        with torch.no_grad():  # Disable gradient calculation for inference
            logits = model(idx_cond)  # Get logits from the model

        # Get the logits for the last position in the sequence
        logits = logits[:, -1, :]  
        
        # Apply top-k filtering if specified
        if top_k is not None:  
            top_logits, _ = torch.topk(logits, top_k)  # Get the top-k logits
            min_val = top_logits[:, -1]  # Get the minimum value among the top-k logits
            # Set logits below the minimum to -inf to effectively mask them out
            logits = torch.where(
                logits < min_val,
                torch.tensor(float('-inf')).to(logits.device),
                logits
            )  

        # Apply temperature-based sampling if specified
        if temperature > 0.0:  
            logits = logits / temperature  # Scale the logits by the temperature
            probs = torch.softmax(logits, dim=-1)  # Apply softmax to get probabilities
            idx_next = torch.multinomial(probs, num_samples=1)  # Sample from the multinomial distribution
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # Select the token with the highest logit (greedy decoding)

        # Stop generation if the end-of-sequence token is generated
        if eos_id is not None and (idx_next == eos_id).any():  
            break

        # Append the next token to the sequence
        idx = torch.cat((idx, idx_next), dim=1)  

    return idx




#%%


def generate_and_print_sample(model, tokenizer, device, start_context, top_k=25, temperature=1.4):
    """
    Generates a text sample using the language model and prints it.

    Args:
      model: The language model to use for text generation.
      tokenizer: The tokenizer object used to convert text to tokens.
      device (torch.device): The device (CPU or GPU) to perform calculations on.
      start_context (str): The initial context used for text generation.
    """
    model.eval()  # Set the model to evaluation mode
    context_size = model.pos_emb.weight.shape[0]  # Get the context size from the model
    # Convert the starting context to token IDs and move to the device
    encoded = text_to_token_ids(start_context, tokenizer).to(device)  
    with torch.no_grad():  # Disable gradient calculation for generation
        # Generate token IDs using the language model
        token_ids = generate(
            model=model, idx=encoded, 
            max_new_tokens=50, context_size=context_size,
            top_k=top_k, temperature=temperature
        )  
    # Decode the generated token IDs back to text
    decoded_text = token_ids_to_text(token_ids, tokenizer)  
    print(decoded_text.replace("\n", " "))  # Print the generated text (removing newlines)
    model.train()  # Set the model back to training mode



# %% Initialize the model
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = tiktoken.get_encoding("gpt2")
model = GPTModel(GPT_CONFIG_124M)
model.eval()



# %%

# Create a list of start_contexts
start_contexts = [
    "Different types of commercial reactors like BWRs and",
    "Canadian research reactors like NRU and ",
    "distinctive features of CANDU reactors such as horizontal fuel ",
    "the nuclear force binding protons and ",
    "how the stability of a nucleus is influenced by the number of ",
    "nuclear decay (alpha, beta "
    # "She looked out the window and saw",
    # "The cat sat on the windowsill and watched",
    # "In the distance, a figure could be seen",
    # "The clock struck midnight and the room was plunged into darkness",
    # "The wind howled through the trees as the storm approached
]

# Generate and print text samples for each start context
for start_context in start_contexts:
    print("\nStart context:", start_context)
    print("Generated text:")
    generate_and_print_sample(model, tokenizer, device, start_context)
    
# %%


# Load checkpoint if available, otherwise train from scratch
checkpoint_path = "model_and_optimizer.pth"
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    print("Loaded model and optimizer from checkpoint.")
    model.eval()
else:
    print("Checkpoint not found.")



# Generate and print text samples for each start context
for start_context in start_contexts:
    print("\nStart context:", start_context)
    print("Generated text:")
    generate_and_print_sample(model, tokenizer, device, start_context, top_k=25, temperature=1.4)
#%%
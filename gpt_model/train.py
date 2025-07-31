

# %% Import model and libraries

import os
import torch
import tiktoken
import math
from torch.utils.data import Dataset, DataLoader
import numpy as np
from model import GPTModel
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# %% Model Configuration


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

# %%



# %%

# The GPTDatasetV1 class is based on the PyTorch Dataset class
# It defines how individual rows are fetched from the dataset,
# where each row consists of a number of token IDs (based on a max_length) assigned
# to an input_chunk tensor. The target_ chunk tensor contains the corresponding targets.



class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(txt)

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1: i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]
    


# %%

# Create a DataLoader instance using the GPTDatasetV1 class
# This DataLoader will be used to iterate over the dataset in batches
# The DataLoader is shuffled and configured to drop the last incomplete batch


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True,
                         num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    )

    return dataloader


# %%

# Defining the loss calculation function
def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Calculates the cross-entropy loss for a single batch of input and target.

    Args:
      input_batch (torch.Tensor): The input batch to the model.
      target_batch (torch.Tensor): The corresponding target batch.
      model: The model used for prediction.
      device (torch.device): The device (CPU or GPU) to perform calculations on.

    Returns:
      torch.Tensor: The calculated cross-entropy loss for the batch.
    """
    input_batch = input_batch.to(device)  # Move input to the specified device
    target_batch = target_batch.to(device)  # Move target to the specified device
    logits = model(input_batch)  # Get model predictions (logits)
    # Calculate cross-entropy loss, flattening the tensors for compatibility
    loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), target_batch.flatten()
    )  
    return loss

# Defining the loss calculation function over data loader
def calc_loss_loader(data_loader, model, device, num_batches=None):
    """
    Calculates the average cross-entropy loss over a data loader.

    Args:
      data_loader (torch.utils.data.DataLoader): The data loader containing 
                                                input and target batches.
      model: The model used for prediction.
      device (torch.device): The device (CPU or GPU) to perform calculations on.
      num_batches (int, optional): The number of batches to process. 
                                   Defaults to None (process all batches).

    Returns:
      float: The average cross-entropy loss over the specified number of batches.
    """
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")  # Return NaN if the data loader is empty
    # Determine the number of batches to process
    elif num_batches is None:
        num_batches = len(data_loader)  # Process all batches if None is specified
    else:
        num_batches = min(num_batches, len(data_loader))  # Process up to num_batches

    # Iterate over the data loader
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )  # Calculate loss for the batch
            total_loss += loss.item()  # Accumulate the loss
        else:
            break  # Stop if the desired number of batches is reached

    return total_loss / num_batches  # Return the average loss

#%%

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """
    Evaluates the language model on the training and validation sets.

    Args:
      model: The language model to evaluate.
      train_loader (torch.utils.data.DataLoader): The data loader for the 
                                                  training data.
      val_loader (torch.utils.data.DataLoader): The data loader for the 
                                                validation 1  data.
      device (torch.device): The device (CPU or GPU) to perform calculations on.
      eval_iter (int): The number of batches to use for evaluation.

    Returns:
      tuple: A tuple containing the average training loss and average 
             validation loss.
    """
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():  # Disable gradient calculation for evaluation
        # Calculate the training loss
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )  
        # Calculate the validation loss
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter
        )  
    model.train()  # Set the model back to training mode
    return train_loss, val_loss  # Return the calculated losses

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


#%%



def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter, start_context, tokenizer,
                warmup_steps, initial_lr=3e-05, min_lr=1e-6):
    """
    Trains a language model with a learning rate schedule and gradient clipping.

    Args:
      model: The language model to train.
      train_loader (torch.utils.data.DataLoader): The data loader for the training data.
      val_loader (torch.utils.data.DataLoader): The data loader for the validation 1  data.
      optimizer: The optimizer used to update the model's parameters.
      device (torch.device): The device (CPU or GPU) to perform calculations on.
      n_epochs (int): The number of epochs to train for.
      eval_freq (int): The frequency (in global steps) of evaluating the model on the validation set.
      eval_iter (int): The number of batches to use for evaluation.
      start_context (str): The initial context used for text generation.
      tokenizer: The tokenizer object used to convert text to tokens.
      warmup_steps (int): The number of warm-up steps for the learning rate schedule.
      initial_lr (float, optional): The initial learning rate. Defaults to 3e-05.
      min_lr (float, optional): The minimum learning rate. Defaults to 1e-6.

    Returns:
      tuple: A tuple containing the training losses, validation losses, the number of 
             tokens seen during training, and the learning rates used during training.
    """
    train_losses, val_losses, track_tokens_seen, track_lrs = [], [], [], []  # Initialize lists to store metrics
    tokens_seen, global_step = 0, -1  # Initialize counters

    peak_lr = optimizer.param_groups[0]["lr"]  # Get the peak learning rate from the optimizer
    total_training_steps = len(train_loader) * n_epochs  # Calculate the total number of training steps
    lr_increment = (peak_lr - initial_lr) / warmup_steps  # Calculate the learning rate increment for warm-up

    for epoch in range(n_epochs):  # Loop over the specified number of epochs
        model.train()  # Set the model to training mode
        for input_batch, target_batch in train_loader:  # Iterate over the training data
            optimizer.zero_grad()  # Reset the gradients
            global_step += 1  # Increment the global step counter

            # Learning Rate Scheduling
            if global_step < warmup_steps:  # Apply linear warm-up
                lr = initial_lr + global_step * lr_increment  
            else:  # Apply cosine decay
                progress = ((global_step - warmup_steps) / 
                            (total_training_steps - warmup_steps))
                lr = min_lr + (peak_lr - min_lr) * 0.5 * (
                    1 + math.cos(math.pi * progress))

            for param_group in optimizer.param_groups:  # Update the learning rate in the optimizer
                param_group["lr"] = lr
            track_lrs.append(lr)  # Store the learning rate

            # Loss Calculation and Backpropagation
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # Calculate the loss for the batch
            loss.backward()  # Backpropagate the loss

            # Gradient Clipping
            if global_step > warmup_steps:  # Apply gradient clipping after warm-up
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=1.0
                )  

            optimizer.step()  # Update the model's parameters
            tokens_seen += input_batch.numel()  # Update the number of tokens seen

            # Evaluation
            if global_step % eval_freq == 0:  # Evaluate the model periodically
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, 
                    device, eval_iter
                )  # Calculate training and validation loss
                train_losses.append(train_loss)  # Store the training loss
                val_losses.append(val_loss)  # Store the validation loss
                track_tokens_seen.append(tokens_seen)  # Store the number of tokens seen
                # Print the training progress
                print(f"Ep {epoch+1} (Iter {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, "
                      f"Val loss {val_loss:.3f}"
                )  

        # Generate and print a text sample after each epoch
        generate_and_print_sample(model, tokenizer, device, start_context)  

    return train_losses, val_losses, track_tokens_seen, track_lrs  # Return the collected metrics




#%% --- Data Loading and Preprocessing ---

def load_data(file_path):
    """
    Loads text data from multiple files in a directory.

    Args:
      file_path (str): The path to the directory containing the text files.

    Returns:
      str: The combined text data from all files.
    """
    text_data = ""
    for filename in os.listdir(file_path):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(file_path, filename), "r", encoding="utf-8") as file:
                    text_data += file.read()
            except Exception as e:
                print(f"Error processing file {filename}: {e}")
    return text_data

def preprocess_data(text_data, train_ratio=0.90):
    """
    Splits the text data into training and validation sets.

    Args:
      text_data (str): The text data to split.
      train_ratio (float, optional): The ratio of data to use for training. 
                                     Defaults to 0.90.

    Returns:
      tuple: A tuple containing the training and validation data.
    """
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]
    return train_data, val_data

# --- Data Loading ---

def create_dataloaders(train_data, val_data):
    """
    Creates PyTorch DataLoader objects for the training and validation data.

    Args:
      train_data (str): The training data.
      val_data (str): The validation data.

    Returns:
      tuple: A tuple containing the training and validation data loaders.
    """
    train_loader = create_dataloader_v1(
        train_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )
    val_loader = create_dataloader_v1(
        val_data,
        batch_size=2,
        max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )
    return train_loader, val_loader


# --- Visualization ---

def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """
    Plots the training and validation losses.

    Args:
      epochs_seen (torch.Tensor): The epochs at which the losses were recorded.
      tokens_seen (list): The number of tokens seen at each evaluation point.
      train_losses (list): The training losses.
      val_losses (list): The validation losses.
    """
    fig, ax1 = plt.subplots(figsize=(5, 3))
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle="-.", label="Validation loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")
    fig.tight_layout()
    plt.show()

#%% --- Main Function ---

def main():
    # Define tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # Load and preprocess data
    file_path = "../candu_book/text_train"
    text_data = load_data(file_path)
    train_data, val_data = preprocess_data(text_data)

    # Create data loaders
    train_loader, val_loader = create_dataloaders(train_data, val_data)

    # Test data loaders (optional)
    print("Train loader:")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader:")
    for x, y in val_loader:
        print(x.shape, y.shape)

    # Initialize model, optimizer, and training parameters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    peak_lr = 5e-4
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)

    n_epochs = 1
    total_steps = len(train_loader) * n_epochs
    warmup_steps = int(0.2 * total_steps)
    print("Warm up Steps:", warmup_steps)

    # Load checkpoint if available, otherwise train from scratch
    checkpoint_path = "model_and_optimizer.pth"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print("Loaded model and optimizer from checkpoint.")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.1)
        print("No checkpoint found. Training from scratch.")
    

    # Train the model
    model.train()
    train_losses, val_losses, tokens_seen, lrs = train_model(
        model, train_loader, val_loader, optimizer, device, n_epochs=n_epochs,
        eval_freq=5, eval_iter=1, start_context="During the fission process, heat",
        tokenizer=tokenizer, warmup_steps=warmup_steps,
        initial_lr=1e-5, min_lr=1e-5
    )

    # Save the model
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        },
        "model_and_optimizer.pth"
    )

    # Visualize training progress
    epochs_tensor = torch.linspace(0, n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    # Generate and print a text sample
    generate_and_print_sample(model, tokenizer, device, "During the fission process, heat")

    # --- Parameter and Size Calculations ---
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params:,}")

    total_params_gpt2 = (
        total_params - sum(p.numel()
        for p in model.out_head.parameters())
    )
    print(f"Number of trainable parameters "
          f"considering weight tying: {total_params_gpt2:,}"
    )
    total_size_bytes = total_params * 4
    total_size_mb = total_size_bytes / (1024 * 1024)
    print(f"Total size of the model: {total_size_mb:.2f} MB")



#%%
if __name__ == "__main__":
    main()



# %%

# Pretraining a GPT-style LLM from Scratch

This project provides a complete implementation for pretraining a decoder-only transformer model, similar to the GPT-2 124M parameter version, from the ground up using PyTorch. It covers all essential components, from the detailed model architecture and custom data loaders to a sophisticated training loop with learning rate scheduling and text generation capabilities.

## Acknowledgement
`Build a Large Language Model (From Scratch)` textbook: This textbook written by Sebastian Raschka serves as guide for learn how to build a transformer based model from scratch

## ✨ Features

* **Custom Model Architecture**: Built from scratch using PyTorch, including:
    * Efficient `MultiHeadAttention` with causal masking.
    * `LayerNorm` and `GELU` activation function implementations.
    * `TransformerBlock` with residual connections and dropout.
    * Complete `GPTModel` integrating token and positional embeddings.
* **Efficient Data Handling**:
    * A custom `torch.utils.data.Dataset` (`GPTDatasetV1`) to process large text files efficiently.
    * A `DataLoader` setup for creating input/target batches for next-token prediction.
* **Advanced Training Loop**:
    * Learning rate scheduler with a **linear warmup** phase followed by **cosine decay**.
    * **Gradient clipping** to ensure training stability.
    * Periodic evaluation on a validation set to monitor for overfitting.
    * **Checkpointing** to save the model and optimizer states, allowing training to be resumed.
* **Text Generation**:
    * A flexible `generate` function to produce new text from a starting context.
    * Supports **greedy decoding**, **temperature sampling**, and **top-k filtering**.
* **Utilities**:
    * Tokenization using OpenAI's `tiktoken` library (for the `gpt2` vocabulary).
    * Functions to plot training and validation loss curves against epochs and tokens seen.

---

## 📂 Project Structure

Of course. Here is the complete README.md content in raw Markdown format. You can copy and paste this directly into a README.md file.



* `model.py`: Contains all the PyTorch `nn.Module` classes that define the GPT architecture, including the attention mechanism, transformer blocks, and the final model.
* `train.py`: The main script for training the model. It handles data loading, model initialization, the training loop, evaluation, and saving the final model.
* `eval.py`: A script for evaluating a trained model or generating text samples. It loads a saved checkpoint for inference.
* `text_train/`: The directory where you should place your raw `.txt` files for training.

---

## 🤖 Model Architecture

The model is a standard decoder-only transformer based on the architecture described in "Attention Is All You Need" and used in models like GPT.

The core components are:
1.  **Token and Positional Embeddings**: Input token IDs are converted to dense vectors, and positional embeddings are added to give the model a sense of sequence order.
2.  **Transformer Blocks**: The model stacks multiple `TransformerBlock` modules. Each block contains:
    * A **Multi-Head Causal Attention** mechanism (`MultiHeadAttention`) that allows each token to attend to previous tokens in the sequence.
    * A position-wise **Feed-Forward Network** (`FeedForward`) with a `GELU` activation function.
    * Both sub-layers use **residual connections** and **Layer Normalization** (`LayerNorm`).
3.  **Output Head**: A final linear layer maps the transformer's output to logits over the vocabulary to predict the next token.

The project is configured to build a **124M parameter model** with the following specifications (`GPT_CONFIG_124M`):
* `vocab_size`: 50,257
* `context_length`: 1024 (or 256 as used in `train.py`)
* `emb_dim`: 768
* `n_layers`: 12
* `n_heads`: 12

---

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3 installed. You'll need the following libraries:

* PyTorch
* tiktoken
* NumPy
* Matplotlib

### Installation

1.  Clone the repository:
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  Install the required packages:
    ```bash
    pip install torch tiktoken numpy matplotlib
    ```

### Data Preparation

1.  Create a directory named `text_train` in the root of the project.
2.  Place all your training data as raw text files (e.g., `my_data.txt`) inside this directory. The `train.py` script will automatically read and combine all `.txt` files in this folder.

### Training the Model

To start training the model from scratch, simply run the `train.py` script.

```bash
python train.py
```

The script will:

1. Load and preprocess the data from the `text_train` directory.

2. Split the data into training (90%) and validation (10%) sets.

3. Initialize the `GPTModel` and the `AdamW` optimizer.

4. Train the model for the specified number of epochs, printing training and validation loss periodically.

5. Generate a text sample after each epoch to show progress.

6. Save the trained model and optimizer state to `model_and_optimizer.pth`.

7. Display a plot of the training and validation loss curves.

If a `model_and_optimizer.pth` checkpoint exists, the script will automatically load it and resume training.

Evaluation and Text Generation
To evaluate the trained model or generate new text, use the `eval.py` script. This script loads the `model_and_optimizer.pth` checkpoint.


```bash
python eval.py
```

You can modify the `start_context` variable and generation parameters (like `temperature` and `top_k`) in the `main` function of `eval.py` to experiment with different outputs.

Example of generated text from a prompt:

Prompt: `During the fission process, heat`

Generated Text: `During the fission process, heat is transferred to the moderator via a second heat exchanger, located in the moderator, a pressure-tube reactor. This section explains the moderator's role in the reactor and the heat exchanger. The moderator is cooled by`
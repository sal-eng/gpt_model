#%%
import torch
import torch.nn as nn
import tiktoken


# %%

## Defining the GPT MODEL CONFIGURATION


# vocab_size refers to a vocabulary of 50,257 words, as used by the BPE tokenizer (see chapter 2).
# context_length denotes the maximum number of input tokens the model can handle via the positional embeddings (see chapter 2).
# emb_dim represents the embedding size, transforming each token into a 768-dimensional vector.
# n_heads indicates the count of attention heads in the multi-head attention mechanism (see chapter 3).
# n_layers specifies the number of transformer blocks in the model, which we will cover in the upcoming discussion.
# drop_rate indicates the intensity of the dropout mechanism (0.1 implies a 10% random drop out of hidden units) to prevent overfitting (see chapter 3).
# qkv_bias determines whether to include a bias vector in the Linear layers of the multi-head attention for query, key, and value computations.


GPT_CONFIG_124M = {
    "vocab_size": 50257,     # Vocabulary size
    "context_length": 1024,  # Context length
    "emb_dim": 768,          # Embedding dimension
    "n_heads": 12,           # Number of attention heads
    "n_layers": 12,          # Number of layers
    "drop_rate": 0.1,        # Dropout rate
    "qkv_bias": False        # Query-Key-Value bias
}


#%%


## An efficient multi-head attention class


class MultiHeadAttention(nn.Module):
    """
    This class implements multi-head causal attention, where the computation 
    is done explicitly over separate attention heads. This is a more 
    efficient implementation compared to using a wrapper with multiple 
    instances of `CausalAttention`.

    Args:
        d_in (int): The input dimension of the tokens.
        d_out (int): The total output dimension of the multi-head attention.
        context_length (int): The maximum length of the sequence.
        dropout (float): The dropout probability.
        num_heads (int): The number of attention heads.
        qkv_bias (bool, optional): Whether to include bias terms in the query, 
                                    key, and value linear transformations. 
                                    Defaults to False.
    """
    def __init__(self, d_in, d_out, context_length, 
                 dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # Calculate dimension of each head

        # Linear transformations for query, key, and value
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)  
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # Linear projection for the output
        self.out_proj = nn.Linear(d_out, d_out)  
        self.dropout = nn.Dropout(dropout)

        # Create a mask to prevent attention to future tokens
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), 
            diagonal=1)
        )

    def forward(self, x):
        """
        Forward pass of the multi-head attention mechanism.

        Args:
            x (torch.Tensor): The input sequence of shape 
                              (batch_size, num_tokens, d_in).

        Returns:
            torch.Tensor: The context vector of shape 
                          (batch_size, num_tokens, d_out).
        """
        b, num_tokens, d_in = x.shape

        # Calculate query, key, and value
        keys = self.W_key(x)  
        queries = self.W_query(x)  
        values = self.W_value(x)  

        # Reshape for multi-head attention
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)  
        values = values.view(b, num_tokens, self.num_heads, self.head_dim) 
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose to get (batch_size, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)  
        queries = queries.transpose(1, 2)  
        values = values.transpose(1, 2)  

        # Calculate attention scores
        attn_scores = queries @ keys.transpose(2, 3)  
        # Apply the mask to prevent attention to future tokens
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]  
        attn_scores.masked_fill_(mask_bool, -torch.inf) 

        # Normalize the scores using softmax
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1  # Scale by square root of key dimension
        )
        attn_weights = self.dropout(attn_weights)  # Apply dropout

        # Calculate the context vector
        context_vec = (attn_weights @ values).transpose(1, 2)  

        # Concatenate the heads and project to the output dimension
        context_vec = context_vec.contiguous().view(
            b, num_tokens, self.d_out
        )  
        context_vec = self.out_proj(context_vec)  
        return context_vec
    




# %% Layer Normalization class

class LayerNorm(nn.Module):
    """
    This class implements Layer Normalization, a normalization technique 
    that normalizes the activations within each layer for each input 
    example independently. This helps to stabilize training and improve 
    convergence speed.

    Args:
        emb_dim (int): The dimension of the input embeddings.
    """
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # A small value to prevent division by zero
        # Learnable parameters for scaling and shifting
        self.scale = nn.Parameter(torch.ones(emb_dim))  
        self.shift = nn.Parameter(torch.zeros(emb_dim))  

    def forward(self, x):
        """
        Forward pass of the Layer Normalization.

        Args:
            x (torch.Tensor): The input tensor of shape 
                              (batch_size, num_tokens, emb_dim).

        Returns:
            torch.Tensor: The normalized tensor with the same shape as input.
        """
        # Calculate the mean and variance along the last dimension
        mean = x.mean(dim=-1, keepdim=True)  
        var = x.var(dim=-1, keepdim=True, unbiased=False)  
        # Normalize the input
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  
        # Scale and shift the normalized input
        return self.scale * norm_x + self.shift
    

#%% GELU activation function

class GELU(nn.Module):
    """
    This class implements the Gaussian Error Linear Unit (GELU) activation 
    function. GELU is a smooth approximation to the rectifier linear unit 
    (ReLU) and has been shown to improve performance in various deep 
    learning models, especially in transformer networks.

    Args:
        None
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """
        Forward pass of the GELU activation function.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying GELU activation.
        """
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))



#%% Feed Forward

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        return self.layers(x)
    

# %% Transformer Block

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"], 
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        return x
# %% GPT MODEL

# Adding TransformerBlock, LayerNorm, Gelu, and FeedForward classes to the GPT model
class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)

        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# %%

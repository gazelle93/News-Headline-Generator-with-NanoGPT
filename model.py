import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Config

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(Config.emb_dim))
        self.bias = nn.Parameter(torch.zeros(Config.emb_dim)) if Config.bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class SingleHeadAttention(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = Config.emb_dim // Config.num_heads
        self.key = nn.Linear(Config.emb_dim, hidden_dim, bias=Config.bias)
        self.query = nn.Linear(Config.emb_dim, hidden_dim, bias=Config.bias)
        self.value = nn.Linear(Config.emb_dim, hidden_dim, bias=Config.bias)
        self.drop = nn.Dropout(Config.dropout)
        
        # This matrix is used as a causal mask so tokens can only attend to themselves and previous tokens, not future tokens.
        # 'register_buffer' ensures it's part of the model state but not a learnable parameter.
        self.register_buffer('tril', torch.tril(torch.ones(Config.block_size, Config.block_size)))
    
    def forward(self, input_emb):
        _, seq_len, emb_dim = input_emb.shape # (batch size, sequence_length, embedding_dim)
        
        k = self.key(input_emb)
        q = self.query(input_emb)
        v = self.value(input_emb)
        
        weights = q @ k.transpose(-2, -1) * emb_dim**(-0.5)
        masked_weights = weights.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf"))
        masked_probs = F.softmax(masked_weights, dim=-1)
        masked_probs = self.drop(masked_probs)
        output = masked_probs @ v

        return output
        
class MultiHeadSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        hidden_dim = Config.emb_dim // Config.num_heads
        self.heads = nn.ModuleList([SingleHeadAttention() for _ in range(Config.num_heads)])
        self.project = nn.Linear(Config.emb_dim, Config.emb_dim)
        self.drop = nn.Dropout(Config.dropout)
    
    def forward(self, input_emb):
        output = torch.cat([sh(input_emb) for sh in self.heads], dim=-1)
        output = self.project(output)
        output = self.drop(output)
        return output
    
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        """
        Expanding to a larger hidden size:
         - The network can represent more patterns.
         - It can learn richer combinations of features.
        """
        self.layer = nn.Sequential(
            nn.Linear(Config.emb_dim, Config.dim_expansion*Config.emb_dim), 
            nn.ReLU(),
            nn.Linear(Config.dim_expansion*Config.emb_dim, Config.emb_dim), 
            nn.Dropout(Config.dropout)
        )

    def forward(self, input_emb):
        return self.layer(input_emb)
    
class MoEFeedForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_experts = Config.num_experts
        self.top_k = Config.top_k
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(Config.emb_dim, Config.dim_expansion * Config.emb_dim),
                nn.ReLU(),
                nn.Linear(Config.dim_expansion * Config.emb_dim, Config.emb_dim),
                nn.Dropout(Config.dropout)
            ) for _ in range(Config.num_experts)
        ])
        
        # It learns to assign weights to each expert for each token.
        self.router = nn.Linear(Config.emb_dim, Config.num_experts)

    def forward(self, input_emb):
        # x: (batch, seq, embed)
        batch_size, seq_len, emb_dim = input_emb.shape
        route_logits = self.router(input_emb)  # (batch, seq, num_experts)
        route_weights = F.softmax(route_logits, dim=-1)  # probabilities

        # Top-k routing (zero-out all but top-k)
        topk_vals, topk_idx = torch.topk(route_weights, self.top_k, dim=-1)
        mask = torch.zeros_like(route_weights)
        mask.scatter_(-1, topk_idx, 1.0)
        routed_weights = route_weights * mask  # zero-out all but top-k
        routed_weights = routed_weights / routed_weights.sum(dim=-1, keepdim=True)  # renormalize

        # Compute each expert output
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(input_emb))  # (batch, seq, embed) for each

        # Stack: (num_experts, batch, seq, embed)
        expert_outputs = torch.stack(expert_outputs, dim=0)

        # Weighted sum over experts
        routed_weights = routed_weights.permute(2, 0, 1).unsqueeze(-1)  # (num_experts, batch, seq, 1)
        output = (routed_weights * expert_outputs).sum(dim=0)  # (batch, seq, embed)

        return output
        
class TransformerBlock(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm_1 = LayerNorm()
        self.attn = MultiHeadSelfAttention()
        self.norm_2 = LayerNorm()
        if Config.isMoe:
            self.feedforward = MoEFeedForward()
        else:
            self.feedforward = MLP()

    def forward(self, input_emb):
        # Pre-LN (LayerNorm before each sub-layer)
        normed_output_1 = self.norm_1(input_emb)
        attention_output = self.attn(normed_output_1)
        residual_connection_output_1 = input_emb + attention_output
        
        normed_output_2 = self.norm_2(residual_connection_output_1)
        feedforward_output = self.feedforward(normed_output_2)
        final_output = residual_connection_output_1 + feedforward_output
        return final_output

class GPT(nn.Module):
    def __init__(self):
        super().__init__()
        # Token embeddings lookup table
        self.embedding = nn.Embedding(Config.vocab_size, Config.emb_dim)
        # Positional embeddings lookup table
        self.positional_embedding_table = nn.Embedding(Config.block_size, Config.emb_dim)
        # Stack of Transformer blocks
        self.blocks = nn.Sequential(
            *[TransformerBlock() for _ in range(Config.num_layers)],
        )
        # Final layer normalization before output projection
        self.norm = nn.LayerNorm(Config.emb_dim)        
        # Output linear layer to produce logits for each token
        self.fc = nn.Linear(Config.emb_dim, Config.vocab_size)
    
    def forward(self, input_emb):
        batch_size, seq_len = input_emb.shape
        # Get token embeddings for input indices
        token_embeddings = self.embedding(input_emb)  # (batch_size, seq_len, emb_dim)
        # Get positional embeddings and add to token embeddings
        positional_embedding = self.positional_embedding_table(torch.arange(seq_len, device=input_emb.device))  # (seq_len, emb_dim)
        input_emb = token_embeddings + positional_embedding  # broadcasting (batch_size, seq_len, emb_dim)
        # Pass through Transformer blocks
        transformer_block_output = self.blocks(input_emb)
        # Normalize before output
        normed_output = self.norm(transformer_block_output)
        # Project to vocabulary logits
        logits = self.fc(normed_output)  # (batch_size, seq_len, vocab_size)
        # Flatten batch and time dims for loss computation convenience
        logits = logits.reshape(batch_size * seq_len, Config.vocab_size)
        return logits

    def generate(self, idx, max_tokens):
        """
        Generate tokens autoregressively given a starting context idx.
        """
        for _ in range(max_tokens):
            # Crop idx to block_size (context window)
            idx_cond = idx[:, -Config.block_size:]
            # Forward pass to get logits
            logits = self.forward(idx_cond)  # (batch_size*seq_len, vocab_size)
            batch_size, seq_len = idx_cond.shape
            logits = logits.view(batch_size, seq_len, Config.vocab_size)
            logits = logits[:, -1, :]  # Take logits for last time step
            probs = F.softmax(logits, dim=-1)  # Probability distribution over vocab
            # Sample from distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # Append generated token to sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

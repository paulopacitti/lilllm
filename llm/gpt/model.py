from torch import nn
import torch


class SelfAttention(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.T
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        z = attention_weights @ values  # context vector
        return z


class CausalAttention(nn.Module):
    mask: torch.Tensor

    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        queries = self.W_query(x)
        keys = self.W_key(x)
        values = self.W_value(x)
        attention_scores = queries @ keys.transpose(1, 2)
        attention_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf
        )
        attention_weights = torch.softmax(
            attention_scores / keys.shape[-1] ** 0.5, dim=-1
        )
        attention_weights = self.dropout(attention_weights)

        z = attention_weights @ values
        return z


class GPT(nn.Module):
    context_length: int
    token_embedding_layer: nn.Embedding
    positional_embedding_layer: nn.Embedding

    def __init__(self, context_length: int):
        super().__init__()
        output_dim = 256
        vocab_size = 50257
        self.context_length = context_length  # type: ignore[assignment]
        self.token_embedding_layer = nn.Embedding(vocab_size, output_dim)
        self.positional_embedding_layer = nn.Embedding(context_length, output_dim)

    def forward(self, x):
        tok_embeddings = self.token_embedding_layer(x)
        pos_embeddings = self.positional_embedding_layer(
            torch.arange(self.context_length)
        )
        x = tok_embeddings + pos_embeddings
        return x

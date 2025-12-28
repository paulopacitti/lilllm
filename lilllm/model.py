from torch import nn
import torch


class LilLLM(nn.Module):
    def __init__(self, context_length: int):
        super().__init__()
        output_dim = 256
        vocab_size = 50257
        self.context_length = context_length
        self.token_embedding_layer = nn.Embedding(vocab_size, output_dim)
        self.positional_embedding_layer = nn.Embedding(context_length, output_dim)

    def forward(self, x):
        tok_embeddings = self.token_embedding_layer(x)
        pos_embeddings = self.positional_embedding_layer(
            torch.arange(self.context_length)
        )
        x = tok_embeddings + pos_embeddings
        return x

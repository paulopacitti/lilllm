import torch
from llm.gpt.model import GPT
from llm.gpt.run import generate_text
from llm.gpt.tokenizer import Tokenizer


torch.manual_seed(1234)

tokenizer = Tokenizer("gpt2")
start_context = "Hello, I am"
encoded = tokenizer.encode(start_context)
print("encoded:", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0)
print("encoded_tensor.shape:", encoded_tensor.shape)

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

model = GPT(GPT_CONFIG_124M)
model.eval()
out = generate_text(
    model,
    idx=encoded_tensor,
    max_output_tokens=6,
    context_size=GPT_CONFIG_124M["context_length"],
)

print("Output:", out)
print("Output length:", len(out[0]))
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print(decoded_text)

from os import path
from lilllm.gpt import data
from lilllm.gpt.model import GPT
import torch

DATASET = path.join(path.dirname(__file__), "./data/the-verdict.txt")


def main():
    # load dataset and dataloader
    max_length = 4
    dataset = data.load_pretraining_dataset(DATASET)
    dataloader = data.build_pretraining_dataloader(
        dataset, batch_size=8, max_length=max_length, stride=4, shuffle=True
    )
    dataloader_iter = iter(dataloader)
    inputs, targets = next(dataloader_iter)
    print(f"[dataloader] inputs:\n{inputs}")
    print(f"[dataloader] inputs.shape: {inputs.shape}")

    # embedding layer
    output_dim = 256
    vocab_size = 50257
    embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
    token_embeddings = embedding_layer(inputs)
    print(f"[embedding_layer] token_embeddings.shape: {token_embeddings.shape}")
    # positional embeddings
    context_length = max_length
    positional_embedding_layer = torch.nn.Embedding(context_length, output_dim)
    positional_embeddings = positional_embedding_layer(torch.arange(context_length))
    print(
        f"[positional_embedding_layer] positional_embedding_layer.shape: {positional_embeddings.shape}"
    )
    # input embeddings
    input_embeddings = token_embeddings + positional_embeddings
    print(f"[input_embeddings] input_embeddings.shape: {input_embeddings.shape}")

    # model embeddings
    model = GPT(max_length)
    output = model(inputs)
    print(f"[output]: output:\n{output}")


if __name__ == "__main__":
    main()

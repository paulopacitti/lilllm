from os import path
from lilllm import data

DATASET = path.join(path.dirname(__file__), "./data/the-verdict.txt")


def main():
    with open(DATASET, "r", encoding="utf-8") as file:
        raw_text = file.read()

    dataloader = data.build_pretraining_dataloader(
        raw_text, batch_size=8, max_length=4, stride=4, shuffle=True
    )
    dataloader_iter = iter(dataloader)
    inputs, targets = next(dataloader_iter)
    print(f"inputs: {inputs}")
    print(f"targets: {targets}")


if __name__ == "__main__":
    main()

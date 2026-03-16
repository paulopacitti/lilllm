# AGENTS.md

## Project Overview

LLM implementations and experiments library. Based on "Build a Large Language Model (from scratch)" by Sebastian Raschka.

## Project Structure

```
llm/                     # Library package
├── __init__.py
└── gpt/                 # GPT-related modules
    ├── model.py         # Neural network architectures
    ├── data.py          # Dataset and dataloader utilities
    └── tokenizer.py    # Tokenization utilities

experiments/             # Experiment scripts
├── main.py              # Example: run with `PYTHONPATH=. uv run python experiments/main.py`
└── ...

data/                    # Data files (outside library)
```

## Build/Run Commands

```bash
# Install dependencies
uv sync

# Run experiment scripts (PYTHONPATH needed for subdirectories)
PYTHONPATH=. uv run python experiments/main.py

# Type checking (linter)
uv run ty check

# Run Python REPL
uv run python
```

## Code Style

- Use assertions for internal invariants

### Type Hints

- Add type hints to function signatures
- Return types are encouraged but optional for obvious returns

```python
def encode(self, text: str) -> list[int]:
    ...

def forward(self, x: Tensor) -> Tensor:
    ...
```

## Adding New Components

1. New architectures → `llm/gpt/model.py` or new submodule
2. New data processing → `llm/gpt/data.py` or new submodule
3. New experiments → `experiments/` directory

## Notes

- Use relative imports within the library (`from .tokenizer import Tokenizer`)
- Data files reference from experiments using relative paths: `../data/filename`
- Tokenizer uses `tiktoken` with `gpt2` encoding by default; implement from scratch later

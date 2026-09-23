import mlx.core as mx

# To test independently:
# uv run python -c 'from python_mlx_llm.tokenizer import CharacterTokenizer; from python_mlx_llm.data import create_training_windows; t = CharacterTokenizer("hello mlx"); ids = t.encode("hello mlx", add_end_token=True); x, y = create_training_windows(ids, 3); print(x.shape, y.shape); print(x[0].tolist(), y[0].tolist())'
#
# Expected:
# (7, 3) (7, 3)
# [2, 1, 3] [1, 3, 3]
def create_training_windows(
    token_ids: list[int],
    context_size: int,
) -> tuple[mx.array, mx.array]:

    if context_size <= 0:
        raise ValueError("context_size must be positive")

    if len(token_ids) <= context_size:
        raise ValueError("The token sequence my be longer than context_size")

    tokens = mx.array(token_ids)

    input_examples = []
    target_examples = []

    for start in range(len(tokens) - context_size):
        context = tokens[
            start : start + context_size
        ]

        targets = tokens[
            start + 1 : start + context_size + 1
        ]

        input_examples.append(context)
        target_examples.append(targets)

    inputs = mx.stack(input_examples)
    targets = mx.stack(target_examples)

    return inputs, targets

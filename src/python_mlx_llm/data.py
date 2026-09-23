import mlx.core as mx
from collections.abc import Iterator

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

def iterator_batches(
    inputs: mx.array,
    targets: mx.array,
    batch_size: int,
) -> Iterator[tuple[mx.array, mx.array]]:

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    if inputs.shape[0] != targets.shape[0]:
        raise ValueError("inputs and targets must contain the same number of examples")


    example_count = inputs.shape[0]
    shuffled_indices = mx.random.permutation(example_count)

    for start in range(0, example_count, batch_size):
        batch_indices = shuffled_indices[
            start : start + batch_size
        ]

        batch_inputs = inputs[batch_indices]
        batch_targets = targets[batch_indices]

        yield batch_inputs, batch_targets

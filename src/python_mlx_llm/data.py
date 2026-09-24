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

# Independent test:
# uv run python -c 'import mlx.core as mx; from python_mlx_llm.tokenizer import CharacterTokenizer; from python_mlx_llm.data import sample_batch; mx.random.seed(42); t = CharacterTokenizer("hello mlx"); ids = t.encode("hello mlx", add_end_token=True); x, y = sample_batch(ids, context_size=3, batch_size=4); print(x.shape, y.shape); print(x.tolist()); print(y.tolist())'
#
# Expected shape
# (4, 3) (4, 3)
def sample_batch(
    token_ids: list[int],
    context_size: int,
    batch_size: int,
) -> tuple[mx.array, mx.array]:

    if context_size <= 0:
        raise ValueError("context_size must be positive")

    if batch_size <= 0:
        raise ValueError("batch_size must be positve")

    if len(token_ids) <= context_size:
        raise ValueError("The token sequence must be larger than context_size")

    tokens = mx.array(token_ids)

    start_indices = mx.random.randint(
        low=0,
        high=len(token_ids) - context_size,
        shape=(batch_size,),
    )

    input_examples = []
    target_examples = []

    for start in start_indices.tolist():
        inputs = tokens[start : start + context_size]

        targets = tokens[start + 1 : start + context_size + 1]

        input_examples.append(inputs)
        target_examples.append(targets)

    input_batch = mx.stack(input_examples)
    target_batch = mx.stack(target_examples)

    return input_batch, target_batch

def iterate_batches(
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

# to test:
# uv run python -c 'from python_mlx_llm.data import split_token_sequence; train, val = split_token_sequence(list(range(10)), 0.8); print(train); print(val)'
#
# Result
#
# [0, 1, 2, 3, 4, 5, 6, 7]
# [8, 9]
def split_token_sequence(
    token_ids: list[int],
    training_fraction: float = 0.9,
) -> tuple[list[int], list[int]]:

    if not 0.0 < training_fraction < 1.0:
        raise valueError("training_fraction must be between 0 and 1")

    split_index = int(len(token_ids) * training_fraction)

    training_tokens = token_ids[:split_index]
    validation_tokens = token_ids[split_index:]

    if not training_tokens or not validation_tokens:
        raise ValueError("Both splits must contain at least one token")

    return training_tokens, validation_tokens

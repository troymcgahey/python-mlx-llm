import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from python_mlx_llm.tokenizer import CharacterTokenizer
from python_mlx_llm.data import (
    create_training_windows,
    iterator_batches,
)
from python_mlx_llm.model import ContextLanguageModel
from path import Path


def loss_fn(
    model: ContextLanguageModel,
    inputs: mx.array,
    targets: mx.array,
) -> mx.array:
    # Compare each position's vocabulary scores with its correct next-token ID.
    # Mean cross-entropy is the single number that training tries to reduce.
    logits = model(inputs)

    return nn.losses.cross_entropy(
        logits,
        targets,
        reduction="mean",
    )

def main() -> None:
    # The toy corpus includes a special token meaning "stop generating."
    training_path = Path("data/input.txt")
    training_text = training_path.read_text(
        encoding="utf-8"
    )

    print(
        "Loaded",
        len(training_text),
        "characters from",
        training_path,
    )

    end_token = "<EOS>"
    context_size = 3

    tokenizer = CharacterTokenizer(
        training_text=training_text,
        end_token=end_token,
    )

    tokenized_text = tokenizer.encode(
        training_text,
        add_end_token=True,
    )

    tokens = mx.array(tokenized_text)

    char_to_id = tokenizer.token_to_id
    id_to_char = tokenizer.id_to_token

    inputs, targets = create_training_windows(
        token_ids=tokenized_text,
        context_size=context_size,
    )

    mx.random.seed(42)

    for batch_inputs, batch_targets in iterator_batches(
        inputs,
        targets,
        batch_size=4,
    ):
        print(
            "Mini-batch shapes:",
            batch_inputs.shape,
            batch_targets.shape,
        )

    end_token_id = char_to_id[end_token]

    for example_index in range(inputs.shape[0]):
        context_ids = inputs[example_index].tolist()
        target_ids = targets[example_index].tolist()

        context_text = tokenizer.decode(context_ids)
        target_text = tokenizer.decode(target_ids)

        print(
            "Context:",
            repr(context_text),
            "Target:",
            repr(target_text),
        )

    # Build a small model with eight learned features per token/position.
    model = ContextLanguageModel(
        vocabulary_size=tokenizer.vocabulary_size,
        context_size=context_size,
        embedding_size=8,
    )

    logits = model(inputs)
    mx.eval(logits)

    # Automatic differentiation computes how each weight affects the loss.
    # SGD uses those gradients to adjust weights after every step.
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=0.5)

    # Reuse the same tiny batch so we can observe the model memorizing it.
    for step in range(1001):
        loss, gradients = loss_and_grad_fn(
            model,
            inputs,
            targets,
        )

        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)

        if step % 20 == 0:
            print("Step:", step, "Loss:", loss.item())

    # Probe the first position. The causal mask lets it see only "l", not the
    # later tokens, so both observed first-position continuations remain valid.
    letter_l_id = char_to_id["l"]
    letter_o_id = char_to_id["o"]

    probe_input = mx.array(
        [[letter_l_id, letter_l_id, letter_o_id]]
    )

    probe_logits = model(probe_input)
    first_position_logits = probe_logits[0, 0]
    first_position_probabilities = mx.softmax(
        first_position_logits
    )

    print(
        "P(next='l' | current='l'):",
        first_position_probabilities[letter_l_id].item(),
    )

    print(
        "P(next='o' | current='l'):",
        first_position_probabilities[letter_o_id].item(),
    )

    # Inspect the 3x3 attention matrix for one example after training.
    inspection_text = "hel"
    inspection_ids = []

    for character in inspection_text:
        inspection_ids.append(char_to_id[character])

    inspection_input = mx.array([inspection_ids])

    inspection_logits, inspection_attention = model(
        inspection_input,
        return_attention=True,
    )

    attention_matrix = inspection_attention[0].tolist()

    print("Attention columns:", list(inspection_text))

    for position, row in enumerate(attention_matrix):
        query_character = inspection_text[position]

        print(
            "Attention from",
            repr(query_character),
            ":",
            row,
        )


    # Generate autoregressively: predict, append, then slide to the newest
    # three token IDs. Stop when the model predicts <EOS>.
    prompt = "hel"

    generated_ids = []

    for character in prompt:
        generated_ids.append(char_to_id[character])

    for step in range(12):
        context_ids = generated_ids[-context_size:]
        context_input = mx.array([context_ids])

        # [0, -1] selects the final position of the only batch example.
        next_logits = model(context_input)[0, -1]
        next_id = mx.argmax(next_logits).item()

        context_characters = []

        for tokenized_text in context_ids:
            context_characters.append(id_to_char[tokenized_text])

        context_text = "".join(context_characters)
        next_character = id_to_char[next_id]

        print(
            "Generation step:",
            step,
            "Context",
            repr(context_text),
            "Prediction",
            repr(next_character),
        )

        if next_id == end_token_id:
            print("Reached the end-of-sequence token.")
            break

        generated_ids.append(next_id)

    # Translate generated IDs into text for display.
    generated_characters = []

    for token in generated_ids:
        generated_characters.append(id_to_char[token])

    generated_text = "".join(generated_characters)

    print("Context model generated:", repr(generated_text))

    print("Model input shape:", inputs.shape)
    print("Model output shape:", logits.shape)

    print("Input batch:")
    print(inputs)
    print("Input shape:", inputs.shape)

    print("Target batch:")
    print(targets)
    print("Target shape:", targets.shape)

if __name__ == "__main__":
    main()
    

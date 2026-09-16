import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

def loss_fn(
    model: nn.Module,
    inputs: mx.array,
    targets: mx.array,
) -> mx.array:
    logits = model(inputs)

    return nn.losses.cross_entropy(
        logits,
        targets,
        reduction="mean",
    )

def main() -> None:

    text = "hello mlx"

    characters = sorted(set(text))

    char_to_id = {}
    for index, character in enumerate(characters):
        char_to_id[character] = index

    id_to_char = {}
    for character, index in char_to_id.items():
        id_to_char[index] = character


    token_ids = []
    for character in text:
        token_id = char_to_id[character]
        token_ids.append(token_id)
    tokens = mx.array(token_ids)

    inputs = tokens[:-1]
    targets = tokens[1:]

    counts = mx.zeros(
        shape=(len(characters), len(characters)),
        dtype=mx.int32
    )

    for position in range(len(inputs)):
        input_id = inputs[position].item()
        target_id = targets[position].item()

        counts = counts.at[input_id, target_id]. add(1)

        input_character = id_to_char[input_id]
        target_character = id_to_char[target_id]

        print(
            "Training pair:",
            repr(input_character),
            "->",
            repr(target_character),
        )

    smoothed_counts = counts + 1

    row_totals = mx.sum(
        smoothed_counts,
        axis=1,
        keepdims=True,
    )

    probabilities = smoothed_counts / row_totals

    mx.random.seed(42)

    current_id = char_to_id["h"]
    generated_characters = [id_to_char[current_id]]

    for step in range(30):
        next_token_probabilities = probabilities[current_id]
        logits = mx.log(next_token_probabilities)

        next_id = mx.random.categorical(logits).item()
        next_character = id_to_char[next_id]

        generated_characters.append(next_character)
        current_id = next_id

    generated_text = "".join(generated_characters)

    print("Generated text:", repr(generated_text))

    letter_l_id = char_to_id["l"]

    print(
        "Probabilities after 'l':",
        probabilities[letter_l_id].tolist(),
    )

    print(
        "Probability row totals:",
        mx.sum(probabilities, axis=1).tolist(),
    )

    print("Transition counts:")
    
    column_labels = []

    for character in characters:
        column_labels.append(repr(character))

    print("Next-token columns:", column_labels)

    for input_id in range(len(characters)):
        input_character = id_to_char[input_id]
        row = counts[input_id].tolist()
        print("After", repr(input_character), ":", row)

    print("Inputs:", inputs)
    print("Targets:", targets)

    decoded_characters = []
    for token in tokens:
        token_id = token.item()
        character = id_to_char[token_id]
        decoded_characters.append(character)

    decoded = "".join(decoded_characters)

    print("Vocabulary:", characters)
    print("Vocabulary size:", len(characters))
    print("Encoded:", tokens)
    print("Decoded:", decoded)
    print("Decoded characer list:", decoded_characters)

    ################### NEW 09/02/2026  #####################3

    vocabulary_size = len(characters)

    model = nn.Embedding(
        num_embeddings=vocabulary_size,
        dims=vocabulary_size,
    )

    logits = model(inputs)
    mx.eval(logits)

    print("Input shape:", inputs.shape)
    print("Logits shape:", logits.shape)
    print("Scores for the first input:", logits[0].tolist())

    first_logits = logits[0]
    first_probabilities = mx.softmax(first_logits)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    optimizer = optim.SGD(learning_rate=0.5)

    for step in range(101):
        loss, gradients = loss_and_grad_fn(
            model,
            inputs,
            targets,
        )

        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)

        if step % 10 == 0:
            print("Step:", step, "Loss:", loss.item())

    letter_l_id = char_to_id["l"]
    letter_l_input = mx.array([letter_l_id])

    trained_logits = model(letter_l_input)
    trained_probabilities = mx.softmax(trained_logits[0])

    print("Next-token columns:", column_labels)
    print(
        "Learned probabilities after 'l':",
        trained_probabilities.tolist(),
    )



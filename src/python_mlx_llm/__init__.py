import mlx.core as mx

def main() -> None:

    text = "hello mlx"

    characters = sorted(set(text))

    char_to_id = {}
    for index, character in enumerate(characters):
        print("Assigning", repr(character), "to token", index)
        char_to_id[character] = index

    id_to_char = {}
    for character, index in char_to_id.items():
        id_to_char[index] = character

    print("Character to ID:", char_to_id)
    print("ID to character:", id_to_char)

    #tokens = mx.array([char_to_id[character] for character in text])
    token_ids = []
    for character in text:
        token_id = char_to_id[character]
        token_ids.append(token_id)
    print("Token ID list:", token_ids)
    tokens = mx.array(token_ids)
    #decoded = "".join(id_to_char[token.item()] for token in tokens)

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

import mlx.core as mx

def main() -> None:

    text = "hello mlx"

    characters = sorted(set(text))
    #char_to_id = {character: index for index, character in enumerate(characters)}
    char_to_id = {}
    for index, character in enumerate(characters):
        print("Assigning", repr(character), "to token", index)
        char_to_id[character] = index

    id_to_char = {index: character for character, index in char_to_id.items()}

    #tokens = mx.array([char_to_id[character] for character in text])
    token_ids = []
    for character in text:
        token_id = char_to_id[character]
        token_ids.append(token_id)
    print("Token ID list:", token_ids)
    tokens = mx.array(token_ids)
    decoded = "".join(id_to_char[token.item()] for token in tokens)

    print("Vocabulary:", characters)
    print("Vocabulary size:", len(characters))
    print("Encoded:", tokens)
    print("Decoded:", decoded)

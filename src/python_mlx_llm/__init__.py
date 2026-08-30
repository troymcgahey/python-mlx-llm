import mlx.core as mx

def main() -> None:

    text = "hello mlx"

    #creates a set of characters in order from text.
    #creating a set will also eliminate duplicates
    characters = sorted(set(text))

    #creates a dictionary mapping characters to index value
    #char_to_id = {character: index for index, character in enumerate(characters)}
    char_to_id = {}
    for index, character in enumerate(characters):
        print("Assigning", repr(character), "to token", index)
        char_to_id[character] = index

    #creates a second dictionary mapping tokens to characters
    #id_to_char = {index: character for character, index in char_to_id.items()}
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

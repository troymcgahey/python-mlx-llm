import mlx.core as mx

def main() -> None:
    text = "hello mlx"
    context_size = 3

    characters = sorted(set(text))

    char_to_id = {}

    for index, character in enumerate(characters):
        char_to_id[character] = index

    id_to_char = {}

    for character, index in char_to_id.items():
        id_to_char[index] = character

    token_ids = []

    for character in text:
        token_ids.append(char_to_id[character])

    tokens = mx.array(token_ids)

    for start in range(len(tokens) - context_size):
        context = tokens[start : start + context_size]
        target = tokens[start + context_size]
        
        context_characters = []

        for token in context:
            context_characters.append(id_to_char[token.item()])

        context_text = "".join(context_characters)
        target_character = id_to_char[target.item()]

        print(
            "Context:",
            repr(context_text),
            "Target:",
            repr(target_character),
        )

if __name__ == "__main__":
    main()
    

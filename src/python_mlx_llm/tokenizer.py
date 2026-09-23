# To test this class indepently:
# uv run python -c 'from python_mlx_llm.tokenizer import CharacterTokenizer; t = CharacterTokenizer("hello mlx"); ids = t.encode("hello mlx", add_end_token=True); print(ids); print(t.decode(ids)); print(t.vocabulary_size)'
#
# Output:
# [2, 1, 3, 3, 5, 0, 4, 3, 6, 7]
# hello mlx<EOS>
# 8
class CharacterTokenizer:
    def __init__(
        self,
        training_text: str,
        end_token: str = "<EOS>",
    ) -> None:
        self.end_token = end_token

        self.tokens = sorted(set(training_text))
        self.tokens.append(end_token)

        self.token_to_id = {}

        for token_id, token in enumerate(self.tokens):
            self.token_to_id[token] = token_id

        self.id_to_token = {}

        for token, token_id in self.token_to_id.items():
            self.id_to_token[token_id] = token

    @property
    def vocabulary_size(self) -> int:
        return len(self.tokens)

    def encode(
        self,
        text: str,
        add_end_token: bool = False,
    ) -> list[int]:
        token_ids = []

        for character in text:
            token_ids.append(self.token_to_id[character])

        if add_end_token:
            token_ids.append(
                self.token_to_id[self.end_token]
            )

        return token_ids

    def decode(
        self,
        token_ids: list[int],
    ) -> str:
        tokens = []

        for token_id in token_ids:
            tokens.append(self.id_to_token[token_id])

        return "".join(tokens)

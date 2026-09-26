import json
from pathlib import Path

import mlx.core as mx

from python_mlx_llm.model import ContextLanguageModel
from python_mlx_llm.tokenizer import CharacterTokenizer

def main() -> None:
    checkpoint_directory = Path("checkpoints")
    checkpoint_name = "single_head_baseline"

    metadata_path = (
        checkpoint_directory / f"{checkpoint_name}.json"
    )

    weights_path = (
        checkpoint_directory / f"{checkpoint_name}.safetensors"
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    tokenizer = CharacterTokenizer(
        vocabulary=metadata["vocabulary"],
        end_token=metadata["end_token"],
    )

    model = ContextLanguageModel(
        vocabulary_size=tokenizer.vocabulary_size,
        context_size=metadata["context_size"],
        embedding_size=metadata["embedding_size"],
    )

    model.load_weights(str(weights_path))
    model.eval()
    mx.eval(model.parameters())

    prompt = "ROMEO:"
    generation_steps = 300
    temperature = 0.8

    mx.random.seed(42)

    generated_ids = tokenizer.encode(prompt)

    end_token_id = tokenizer.token_to_id[tokenizer.end_token]

    for _ in range(generation_steps):
        context_ids = generated_ids[
            -metadata["context_size"] :
        ]

        context_input = mx.array([context_ids])

        next_logits = model(context_input)[0, -1]
        scaled_logits = next_logits / temperature

        next_id = mx.random.categorical(
            scaled_logits
        ).item()

        if next_id == end_token_id:
            break

        generated_ids.append(next_id)

    generated_text = tokenizer.decode(
        generated_ids 
    )

    print(generated_text)

if __name__ == "__main__":
    main()

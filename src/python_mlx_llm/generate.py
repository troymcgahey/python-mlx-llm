import json
from pathlib import Path
from python_mlx_llm.stacked_model import TransformerLanguageModel
from python_mlx_llm.config import GenerationConfig, ModelConfig

import mlx.core as mx

from python_mlx_llm.model import ContextLanguageModel
from python_mlx_llm.tokenizer import CharacterTokenizer

def main() -> None:
    checkpoint_directory = Path("checkpoints")
    checkpoint_name = "three_layer_transformer"

    metadata_path = (
        checkpoint_directory / f"{checkpoint_name}.json"
    )

    weights_path = (
        checkpoint_directory / f"{checkpoint_name}.safetensors"
    )

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    model_config = ModelConfig.from_dict(metadata)

    tokenizer = CharacterTokenizer(
        vocabulary=metadata["vocabulary"],
        end_token=metadata["end_token"],
    )

    generation_config = GenerationConfig(
        prompt="ROMEO:",
        max_new_tokens=300,
        temperature=0.8,
        random_seed=42,
    )

    model = TransformerLanguageModel(
        vocabulary_size=tokenizer.vocabulary_size,
        context_size=metadata["context_size"],
        embedding_size=model_config.embedding_size,
        num_heads=model_config.num_heads,
        num_layers=model_config.num_layers,
    )

    model.load_weights(str(weights_path))
    mx.eval(model.parameters())

    mx.random.seed(generation_config.random_seed)

    generated_ids = tokenizer.encode(generation_config.prompt)

    end_token_id = tokenizer.token_to_id[tokenizer.end_token]

    for _ in range(generation_config.max_new_tokens):
        context_ids = generated_ids[
            -metadata["context_size"] :
        ]

        context_input = mx.array([context_ids])

        next_logits = model(context_input)[0, -1]
        scaled_logits = next_logits / generation_config.temperature

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

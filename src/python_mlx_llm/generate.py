import json
import argparse
from pathlib import Path
from python_mlx_llm.stacked_model import TransformerLanguageModel
from python_mlx_llm.config import GenerationConfig, ModelConfig

import mlx.core as mx

from python_mlx_llm.model import ContextLanguageModel
from python_mlx_llm.tokenizer import CharacterTokenizer

# uv run python -m python_mlx_llm.generate --help
#
#
#
# uv run python -m python_mlx_llm.generate \
#  --prompt "JULIET:" \
#  --max-new-tokens 150 \
#  --temperature 0.7 \
#  --seed 10
#
#
def parse_arguments() -> argparse.Namespace:
    """Read generation options supplied on the command line."""

    parser = argparse.ArgumentParser(
        description="Generate text using a trained character language model."
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="ROMEO:",
        help="Text the model continues from.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=300,
        help="Maximum number of tokens to generate.",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Sampling randomness: higher values produce more variation.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for reproducible sampling.",
    )

    return parser.parse_args()


def main() -> None:

    arguments = parse_arguments()

    generation_config = GenerationConfig(
        prompt=arguments.prompt,
        max_new_tokens=arguments.max_new_tokens,
        temperature=arguments.temperature,
        random_seed=arguments.seed,
    )

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

from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    """Settings that define the model's architecture."""

    context_size: int = 32
    embedding_size: int=64
    num_heads: int=4
    num_layers: int=3

    def __post_int_(self) -> None:
        if self.context_size <= 0:
            raise ValueError("context_size must be positive")

        if self.embedding_size <= 0:
            raise ValueError("embedding_size must be positive")

        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")

        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")

        if self.embedding_size % self.num_heads != 0:
            raise ValueError("embedding_size must be a multiple of num_heads")


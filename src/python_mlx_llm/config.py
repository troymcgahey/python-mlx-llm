from dataclasses import dataclass

@dataclass(frozen=True)
class ModelConfig:
    """Settings that define the model's architecture."""

    context_size: int = 32
    embedding_size: int = 64
    num_heads: int = 4
    num_layers: int = 3

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


@dataclass(frozen=True)
class TrainingConfig:
    """Settings that control the training process."""

    batch_size: int = 32
    learning_rate: float = 3e-4
    max_steps: int = 3000
    log_interval: int = 100
    evaluation_batches: int = 20
    training_fraction: float = 0.9
    random_seed: int = 42

    def __post_int_(self) -> None:
        
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")

        if self.max_steps <= 0:
            raise ValueError("maxt_steps must be positive")

        if self.log_interval <= 0:
            raise ValueError("log_interval must be positive")

        if self.evaluation_batches <= 0:
            raise ValueError("evaluation_batches must be positive")

        if not 0.0 < self.training_fraction < 1.0:
            raise ValueError("training_fraction must be between 0 and 1")




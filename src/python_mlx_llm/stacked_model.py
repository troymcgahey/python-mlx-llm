import mlx.core as mx
import mlx.nn as nn

class TransformerBlock(nn.Module):
    def __init__(
        self,
        embedding_size: int,
        num_heads: int,
    ) -> None:

        super().__init__()

        if embedding_size % num_heads != 0:
            raise ValueError("embedding size must be a multiple of num_heads")

        self.embedding_size = embedding_size
        self.num_heads = num_heads
        self.head_size = embedding_size // num_heads

        self.attention_norm = nn.LayerNorm(
            dims=embedding_size
        )

        self.query_projection = nn.Linear(
            input_dims=embedding_size,
            output_dims=embedding_size,
        )

        self.key_projection = nn.Linear(
            input_dims=embedding_size,
            output_dims=embedding_size,
        )

        self.value_projection = nn.Linear(
            input_dims=embedding_size,
            output_dims=embedding_size,
        )

        #Mix information from the separate heads after concatenation
        self.attention_output_projection = nn.Linear(
            input_dims=embedding_size,
            output_dims=embedding_size,
        )

        self.feed_forward_norm = nn.LayerNorm(
            dims=embedding_size
        )

        feed_forward_size = embedding_size * 4

        self.feed_forward = nn.Sequential(
            nn.Linear(
                input_dims=embedding_size,
                output_dims=feed_forward_size,
            ),
            nn.GELU(),
            nn.Linear(
                input_dims=feed_forward_size,
                output_dims=embedding_size,
            ),
        )

    def __call__(
        self,
        embeddings: mx.array,
        return_attention: bool=False,
    ):
        batch_size = embeddings.shape[0]
        sequence_length = embeddings.shape[1]

        normalized_embeddings = self.attention_norm(
            embeddings
        )

        queries = self.query_projection(
            normalized_embeddings
        )

        keys = self.key_projection(
            normalized_embeddings
        )

        values = self.value_projection(
            normalized_embeddings
        )

        queries = queries.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size,
        ).transpose(0, 2, 1, 3)

        keys = keys.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size,
        ).transpose(0, 2, 1, 3)

        values = values.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_size,
        ).transpose(0, 2, 1, 3)

        attention_scores = (
            queries @ keys.transpose(0, 1, 3, 2)
        )

        attention_scores = (
            attention_scores / (self.head_size ** 0.5)
        )

        causal_mask = mx.triu(
            mx.full(
                shape=(sequence_length, sequence_length),
                vals=float("-inf"),
            ),
            k=1,
        )

        attention_scores = attention_scores + causal_mask

        attention_weights = mx.softmax(
            attention_scores,
            axis=-1,
        )

        attention_output = (
            attention_weights @ values
        )

        attention_output = attention_output.transpose(
            0,
            2,
            1,
            3,
        ).reshape(
            batch_size,
            sequence_length,
            self.embedding_size,
        )

        attention_output = (
            self.attention_output_projection(
                attention_output
            )
        )

        attended_embeddings = (
            embeddings + attention_output
        )

        normalized_attention = (
            self.feed_forward_norm(
                attended_embeddings
            )
        )

        feed_forward_output = self.feed_forward(
            normalized_attention
        )

        block_output = (
            attended_embeddings
            + feed_forward_output
        )

        if return_attention:
            return block_output, attention_weights

        return block_output

class TransformerLanguageModel(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        context_size: int,
        embedding_size: int,
        num_heads: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        if num_layers <= 0:
            raise ValueError("num_layers must be positive")

        self.context_size = context_size

        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            dims=embedding_size,
        )

        self.position_embedding = nn.Embedding(
            num_embeddings=context_size,
            dims=embedding_size,
        )

        self.blocks = []

        for _ in range(num_layers):
            block = TransformerBlock(
                embedding_size=embedding_size,
                num_heads=num_heads,
            )

            self.blocks.append(block)

        self.output_norm = nn.LayerNorm(
            dims=embedding_size
        )

        self.output = nn.Linear(
            input_dims=embedding_size,
            output_dims=vocabulary_size,
        )

    def __call__(
        self,
        inputs: mx.array,
        return_attention: bool=False,
    ):

        sequence_length = inputs.shape[1]

        if sequence_length > self.context_size:
            raise ValueError("Input sequence is longer than context_size")

        token_embeddings = self.token_embedding(
            inputs
        )

        positions = mx.arange(sequence_length)

        position_embeddings = (
            self.position_embedding(positions)
        )

        hidden_states = (
            token_embeddings + position_embeddings
        )

        all_attention_weights = []

        for block in self.blocks:
            if return_attention:
                hidden_states, attention_weights = block(
                    hidden_states,
                    return_attention=True,
                )

                all_attention_weights.append(
                    attention_weights
                )
            else:
                hidden_states = block(hidden_states)

        hidden_states = self.output_norm(
            hidden_states
        )

        logits = self.output(hidden_states)

        if return_attention:
            return logits, all_attention_weights

        return logits


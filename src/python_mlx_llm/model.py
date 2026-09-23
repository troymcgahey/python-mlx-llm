import mlx.core as mx
import mlx.nn as nn

# A tiny decoder-only language model. Each position predicts the next token
# using only tokens at or before that position.
class ContextLanguageModel(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        context_size: int,
        embedding_size: int,
    ) -> None:
        super().__init__()

        # Keep the sequence length and feature width available during forward passes.
        self.context_size = context_size
        self.embedding_size = embedding_size

        # Turn each token ID into a learned vector of embedding_size features.
        self.token_embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            dims=embedding_size,
        )

        # Give each position its own learned vector so order is visible to the model.
        self.position_embedding = nn.Embedding(
            num_embeddings=context_size,
            dims=embedding_size,
        )

        # Attention compares queries with keys, then mixes the corresponding values.
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

        # Convert each position's features into one score per vocabulary token.
        self.output = nn.Linear(
            input_dims=embedding_size,
            output_dims=vocabulary_size,
        )

        # Normalize features before attention (a pre-normalization block).
        self.attention_norm = nn.LayerNorm(
            dims=embedding_size,
        )

        # The feed-forward network expands features, applies a nonlinearity,
        # then projects back to the original embedding width.
        feed_forward_size = embedding_size * 4

        self.feed_forward_norm = nn.LayerNorm(
            dims=embedding_size,
        )

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

    def __call__(self, inputs: mx.array, 
                 return_attention: 
                 bool = False
    ) -> mx.array:
        # inputs has shape (batch, context_size). Looking up token and position
        # vectors gives compatible arrays of shape (batch, context_size, width)
        # and (context_size, width); MLX broadcasts positions across the batch.
        token_embeddings = self.token_embedding(inputs)

        positions = mx.arange(self.context_size)
        position_embeddings = self.position_embedding(positions)

        embeddings = token_embeddings + position_embeddings

        # Keep the original embeddings for a residual connection while attention
        # reads a normalized version of them.
        normalized_embeddings = self.attention_norm(embeddings)

        # Each position produces a query, key, and value vector.
        queries = self.query_projection(normalized_embeddings)
        keys = self.key_projection(normalized_embeddings)
        values = self.value_projection(normalized_embeddings)

        # Dot products score every query position against every key position.
        # The result has shape (batch, context_size, context_size).
        attention_scores = queries @ keys.transpose(0, 2, 1)
        # Scaling keeps large dot products from making softmax too extreme.
        attention_scores = attention_scores / (self.embedding_size ** 0.5)

        # Put -infinity above the diagonal: a position cannot see future tokens.
        causal_mask = mx.triu(
            mx.full(
                shape=(self.context_size, self.context_size),
                vals=float("-inf"),
            ),
            k=1,
        )

        attention_scores = attention_scores + causal_mask

        # Convert scores to per-row probabilities; masked entries become zero.
        attention_weights = mx.softmax(
            attention_scores,
            axis=-1,
        )

        # Mix value vectors according to attention, then preserve the original
        # embeddings through a residual addition.
        attention_output = attention_weights @ values
        attended_embeddings = embeddings + attention_output

        # Transform each position's features independently, then add another
        # residual connection. This completes one Transformer-style block.
        normalized_attention = self.feed_forward_norm(
            attended_embeddings
        )

        feed_forward_output = self.feed_forward(
            normalized_attention
        )

        transformer_output = (
            attended_embeddings + feed_forward_output
        )

        batch_size = inputs.shape[0]

        # Produce (batch, context_size, vocabulary_size) next-token scores.
        logits = self.output(transformer_output)

        # Optional diagnostic output lets us inspect attention without changing
        # the normal training and generation calls.
        if return_attention:
            return logits, attention_weights

        return logits

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

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
        keys = self.query_projection(normalized_embeddings)
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

def loss_fn(
    model: ContextLanguageModel,
    inputs: mx.array,
    targets: mx.array,
) -> mx.array:
    # Compare each position's vocabulary scores with its correct next-token ID.
    # Mean cross-entropy is the single number that training tries to reduce.
    logits = model(inputs)

    return nn.losses.cross_entropy(
        logits,
        targets,
        reduction="mean",
    )

def main() -> None:
    # The toy corpus includes a special token meaning "stop generating."
    training_text = "hello mlx"
    end_token = "<EOS>"
    context_size = 3

    # The vocabulary assigns one ID to each character plus <EOS>.
    characters = sorted(set(training_text))
    characters.append(end_token)

    # Encode characters as integer token IDs.
    char_to_id = {}

    for index, character in enumerate(characters):
        char_to_id[character] = index

    # Decode token IDs back into readable characters.
    id_to_char = {}

    for character, index in char_to_id.items():
        id_to_char[index] = character

    # Convert the corpus to IDs and append the end marker as its final token.
    tokenized_text = []

    for character in training_text:
        tokenized_text.append(char_to_id[character])

    tokenized_text.append(char_to_id[end_token])

    tokens = mx.array(tokenized_text)

    # Collect overlapping windows. A size-3 input such as "hel" is paired
    # with shifted targets "ell": each position predicts its next token.
    input_examples = []
    target_examples = []

    end_token_id = char_to_id[end_token]

    for start in range(len(tokens) - context_size):
        context = tokens[start : start + context_size]
        target = tokens[
            start + 1 : start + context_size + 1
        ]

        input_examples.append(context)
        target_examples.append(target)
        
        # Decode this training pair only for the human-readable trace.
        context_characters = []

        for token in context:
            context_characters.append(id_to_char[token.item()])

        context_text = "".join(context_characters)
        target_characters = []

        for token in target:
            target_characters.append(id_to_char[token.item()])

        target_text = "".join(target_characters)

        print(
            "Context:",
            repr(context_text),
            "Target:",
            repr(target_text),
        )

    # Stack windows into a batch: inputs and targets are both (7, 3).
    inputs = mx.stack(input_examples)
    targets = mx.stack(target_examples)

    # Build a small model with eight learned features per token/position.
    model = ContextLanguageModel(
        vocabulary_size=len(characters),
        context_size=context_size,
        embedding_size=8,
    )

    logits = model(inputs)
    mx.eval(logits)

    # Automatic differentiation computes how each weight affects the loss.
    # SGD uses those gradients to adjust weights after every step.
    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=0.5)

    # Reuse the same tiny batch so we can observe the model memorizing it.
    for step in range(1001):
        loss, gradients = loss_and_grad_fn(
            model,
            inputs,
            targets,
        )

        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)

        if step % 20 == 0:
            print("Step:", step, "Loss:", loss.item())

    # Probe the first position. The causal mask lets it see only "l", not the
    # later tokens, so both observed first-position continuations remain valid.
    letter_l_id = char_to_id["l"]
    letter_o_id = char_to_id["o"]

    probe_input = mx.array(
        [[letter_l_id, letter_l_id, letter_o_id]]
    )

    probe_logits = model(probe_input)
    first_position_logits = probe_logits[0, 0]
    first_position_probabilities = mx.softmax(
        first_position_logits
    )

    print(
        "P(next='l' | current='l'):",
        first_position_probabilities[letter_l_id].item(),
    )

    print(
        "P(next='o' | current='l'):",
        first_position_probabilities[letter_o_id].item(),
    )

    # Inspect the 3x3 attention matrix for one example after training.
    inspection_text = "hel"
    inspection_ids = []

    for character in inspection_text:
        inspection_ids.append(char_to_id[character])

    inspection_input = mx.array([inspection_ids])

    inspection_logits, inspection_attention = model(
        inspection_input,
        return_attention=True,
    )

    attention_matrix = inspection_attention[0].tolist()

    print("Attention columns:", list(inspection_text))

    for position, row in enumerate(attention_matrix):
        query_character = inspection_text[position]

        print(
            "Attention from",
            repr(query_character),
            ":",
            row,
        )


    # Generate autoregressively: predict, append, then slide to the newest
    # three token IDs. Stop when the model predicts <EOS>.
    prompt = "hel"

    generated_ids = []

    for character in prompt:
        generated_ids.append(char_to_id[character])

    for step in range(12):
        context_ids = generated_ids[-context_size:]
        context_input = mx.array([context_ids])

        # [0, -1] selects the final position of the only batch example.
        next_logits = model(context_input)[0, -1]
        next_id = mx.argmax(next_logits).item()

        context_characters = []

        for tokenized_text in context_ids:
            context_characters.append(id_to_char[tokenized_text])

        context_text = "".join(context_characters)
        next_character = id_to_char[next_id]

        print(
            "Generation step:",
            step,
            "Context",
            repr(context_text),
            "Prediction",
            repr(next_character),
        )

        if next_id == end_token_id:
            print("Reached the end-of-sequence token.")
            break

        generated_ids.append(next_id)

    # Translate generated IDs into text for display.
    generated_characters = []

    for token in generated_ids:
        generated_characters.append(id_to_char[token])

    generated_text = "".join(generated_characters)

    print("Context model generated:", repr(generated_text))

    print("Model input shape:", inputs.shape)
    print("Model output shape:", logits.shape)

    print("Input batch:")
    print(inputs)
    print("Input shape:", inputs.shape)

    print("Target batch:")
    print(targets)
    print("Target shape:", targets.shape)

if __name__ == "__main__":
    main()
    

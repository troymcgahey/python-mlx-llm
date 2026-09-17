import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class ContextLanguageModel(nn.Module):
    def __init__(
        self,
        vocabulary_size: int,
        context_size: int,
        embedding_size: int,
    ) -> None:
        super().__init__()

        self.context_size = context_size
        self.embedding_size = embedding_size

        self.embedding = nn.Embedding(
            num_embeddings=vocabulary_size,
            dims=embedding_size,
        )

        self.output = nn.Linear(
            input_dims=context_size * embedding_size,
            output_dims=vocabulary_size,
        )

    def __call__(self, inputs: mx.array) -> mx.array:
        embeddings = self.embedding(inputs)

        batch_size = inputs.shape[0]

        flattened_embeddings = embeddings.reshape(
            batch_size,
            self.context_size * self.embedding_size,
        )

        logits = self.output(flattened_embeddings)

        return logits

def loss_fn(
    model: ContextLanguageModel,
    inputs: mx.array,
    targets: mx.array,
) -> mx.array:
    logits = model(inputs)

    return nn.losses.cross_entropy(
        logits,
        targets,
        reduction="mean",
    )

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

    input_examples = []
    target_examples = []

    for start in range(len(tokens) - context_size):
        context = tokens[start : start + context_size]
        target = tokens[start + context_size]

        input_examples.append(context)
        target_examples.append(target)
        
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

    inputs = mx.stack(input_examples)
    targets = mx.stack(target_examples)

    model = ContextLanguageModel(
        vocabulary_size=len(characters),
        context_size=context_size,
        embedding_size=8,
    )

    logits = model(inputs)
    mx.eval(logits)

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=0.5)

    for step in range(201):
        loss, gradients = loss_and_grad_fn(
            model,
            inputs,
            targets,
        )

        optimizer.update(model, gradients)
        mx.eval(model.parameters(), optimizer.state, loss)

        if step % 20 == 0:
            print("Step:", step, "Loss:", loss.item())


    prompt = "hel"

    generated_ids = []

    for character in prompt:
        generated_ids.append(char_to_id[character])

    for step in range(12):
        context_ids = generated_ids[-context_size:]
        context_input = mx.array([context_ids])

        next_logits = model(context_input)[0]
        next_id = mx.argmax(next_logits).item()

        generated_ids.append(next_id)

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
    

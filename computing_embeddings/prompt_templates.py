from sentence_transformers import SentenceTransformer

# prompts is an optional argument that accepts a dictionary of prompts with prompt names to prompt texts.
# The prompt will be prepended to the input text during inference.
model = SentenceTransformer(
    "all-MiniLM-L6-v2",
    prompts={
        "classification": "Classify the following text: ",
        "retrieval": "Retrieve semantically similar text: ",
        "clustering": "Identify the topic or theme based on the text: ",
    },
)


embeddings = model.encode("How to bake a strawberry cake", prompt="Retrieve semantically similar text: ")
print(embeddings.shape)

print(embeddings)

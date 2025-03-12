from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

# Encode some text using "binary" quantization
binary_embeddings = model.encode(
    ["I am driving to the lake.", "It is a beautiful day."],
    precision="binary",
)

# Encode some text without quantization & apply quantization afterwards
embeddings = model.encode(["I am driving to the lake.", "It is a beautiful day."])

binary_embeddings = quantize_embeddings(embeddings, precision="binary")

print(embeddings.shape)
print(embeddings.nbytes)
print(embeddings.dtype)
print(binary_embeddings.shape)
print(binary_embeddings.nbytes)
print(binary_embeddings.dtype)
from sentence_transformers import SentenceTransformer, SimilarityFunction

model = SentenceTransformer("all-MiniLM-L6-v2")

# Two lists of sentences about economics
sentences1 = [
    "The inflation rate has grown significantly",
    "The stock market soared today",
    "Consumer confidence is dropping",
]

sentences2 = [
    "Unemployment rates remain stable",
    "Government stimulus impacted spending",
    "Interest rates reached record highs",
]

# Compute embeddings for both lists
embeddings1 = model.encode(sentences1)
embeddings2 = model.encode(sentences2)

# Change the similarity function to DOT distance
model.similarity_fn_name = SimilarityFunction.DOT

# Compute cosine similarities
similarities = model.similarity(embeddings1, embeddings2)

# Output the pairs with their score
for idx_i, sentence1 in enumerate(sentences1):
    print(sentence1)
    for idx_j, sentence2 in enumerate(sentences2):
        print(f" - {sentence2: <50}: {similarities[idx_i][idx_j]:.4f}")
    print()

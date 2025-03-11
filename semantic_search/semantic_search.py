import torch
from sentence_transformers import SentenceTransformer

# Load the pre-trained model
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# Example corpus with economics-related sentences
corpus = [
    "Inflation has been steadily increasing for the last two quarters.",
    "The Federal Reserve is discussing a potential interest rate hike.",
    "Global supply chain issues continue to affect commodity prices.",
    "Unemployment rates have fallen to record lows in several regions.",
    "Consumer spending hit an all-time high during the holiday season.",
    "Several emerging markets are experiencing rapid currency devaluation.",
    "Oil prices surged after geopolitical tensions intensified in the region.",
    "Central banks are exploring digital currencies as a potential solution.",
    "Government stimulus packages boosted the economic recovery."
]

# Convert corpus to embeddings
corpus_embeddings = embedder.encode(corpus, convert_to_tensor=True)

# Example queries
queries = [
    "Why are prices going up everywhere?",
    "Will central banks consider digital currency adoption?",
    "How is consumer behavior changing in the current economy?",
]

# Number of top results to retrieve
top_k = min(5, len(corpus))

for query in queries:
    query_embedding = embedder.encode(query, convert_to_tensor=True)
    
    # Calculate similarity scores between the query and all corpus sentences
    similarity_scores = embedder.similarity(query_embedding, corpus_embeddings)[0]
    scores, indices = torch.topk(similarity_scores, k=top_k)

    print("\nQuery:", query)
    print("Top 5 most similar sentences in the corpus:")
    for score, idx in zip(scores, indices):
        print(f"{corpus[idx]} (Score: {score:.4f})")

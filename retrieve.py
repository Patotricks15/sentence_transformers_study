from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")

docs = [
    "My first paragraph. That contains information",
    "Python is a programming language.",
]
document_embeddings = model.encode(docs)

query = "What is Python?"
query_embedding = model.encode(query)

print(query_embedding)
from sentence_transformers import SentenceTransformer

# 1. Load a pretrained Sentence Transformer model
model = SentenceTransformer("all-MiniLM-L6-v2")

# The sentences to encode
sentences = [
    "Economia é a ciência da alocação dos recursos escassos.",
    "Economia estuda a produção e distribuição dos bens e serviços",
    "Geologia é a ciência que estuda o planeta Terra e seus mecanismos",
]

# 2. Calculate embeddings by calling model.encode()
embeddings = model.encode(sentences)
print(embeddings.shape)
# [3, 384]

# 3. Calculate the embedding similarities
similarities = model.similarity(embeddings, embeddings)
print(similarities)
# tensor([[1.0000, 0.7959, 0.2773],
#         [0.7959, 1.0000, 0.2271],
#         [0.2773, 0.2271, 1.0000]])
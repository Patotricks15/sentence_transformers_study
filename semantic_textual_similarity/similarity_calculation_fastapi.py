from sentence_transformers import SentenceTransformer, SimilarityFunction
from fastapi import FastAPI


model = SentenceTransformer("all-MiniLM-L6-v2")


app = FastAPI()

@app.get("/compute_similarity")
async def compute_similarity(sentences1: str, sentences2: str):
    # Compute embeddings for both lists
    embeddings1 = model.encode(sentences1)
    embeddings2 = model.encode(sentences2)

    # Change the similarity function to DOT distance
    model.similarity_fn_name = SimilarityFunction.DOT

    # Compute cosine similarities
    similarities = model.similarity(embeddings1, embeddings2)
    return {"similarities": similarities[0][0].item()}
from sentence_transformers import SentenceTransformer
from PIL import Image

# Load CLIP model
model = SentenceTransformer("clip-ViT-B-32")

# Encode an image:
img_emb = model.encode(Image.open("samurai_example.jpg"))

# List of text descriptions
phrases = ["Two dogs in the snow", "A cat on a table", "A picture of London at night", "A samurai"]

# Encode text descriptions
text_emb = model.encode(
    phrases
)

# Compute similarities
similarity_scores = model.similarity(img_emb, text_emb)

print(phrases[similarity_scores.argmax()])
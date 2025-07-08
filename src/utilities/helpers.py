import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def create_dataframe(documents):
    """Create a DataFrame from a list of document dictionaries."""
    if not documents:
        return pd.DataFrame()
    return pd.DataFrame(documents)

def calculate_similarity(embedding1, embedding2):
    """Calculate cosine similarity between two embeddings."""
    return cosine_similarity([embedding1], [embedding2])[0][0]

def find_similar_documents(df, target_embedding, top_n=5):
    """Find top_n similar documents based on cosine similarity."""
    df['similarity'] = df['embedding'].apply(lambda x: calculate_similarity(target_embedding, x))
    return df.sort_values(by='similarity', ascending=False).head(top_n)
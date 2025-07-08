import streamlit as st
import joblib
import os
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torch

MODEL_NAME = "ibm-granite/granite-embedding-30m-english"

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the tokenizer and model for embedding generation."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def generate_embedding(text, tokenizer, model, device):
    """Generate embedding for a single text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True,
                       padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embedding.flatten()

def app():
    st.title("Text Classification")

    # Get list of available classification models
    models_dir = "models"
    model_files = [f for f in os.listdir(models_dir) if f.endswith(('.joblib', '.pkl')) and not f.startswith('vectorizer')]

    if not model_files:
        st.warning("No trained classification models found in the 'models' folder. Please train a model first.")
        return

    selected_model_file = st.selectbox("Select a trained classification model", model_files)
    
    # Load the selected classification model
    try:
        model_path = os.path.join(models_dir, selected_model_file)
        classification_model = joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading classification model: {e}")
        return

    # Load the embedding model (tokenizer and model)
    try:
        tokenizer, embedding_model = load_embedding_model()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        embedding_model.to(device)
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        return

    st.subheader("Enter Text for Classification")
    input_method = st.radio("Choose input method", ("Single Text", "Multiple Texts (new line separated)"))

    texts_to_classify = []
    if input_method == "Single Text":
        text_input = st.text_area("Enter text here")
        if text_input:
            texts_to_classify.append(text_input)
    else: # Multiple Texts
        multi_text_input = st.text_area("Enter multiple texts, each on a new line")
        if multi_text_input:
            texts_to_classify = [t.strip() for t in multi_text_input.split('\n') if t.strip()]

    if st.button("Classify"):
        if texts_to_classify:
            try:
                # Generate embeddings for input texts
                embeddings = []
                for text in texts_to_classify:
                    embeddings.append(generate_embedding(text, tokenizer, embedding_model, device))
                
                # Predict using the classification model
                predictions = classification_model.predict(embeddings)
                
                st.subheader("Classification Results:")
                results_df = pd.DataFrame({
                    "Input Text": texts_to_classify,
                    "Predicted Class": predictions
                })
                st.table(results_df)

            except Exception as e:
                st.error(f"Error during classification: {e}")
        else:
            st.warning("Please enter some text to classify.")

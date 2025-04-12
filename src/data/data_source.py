import streamlit as st
import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm

MODEL_NAME = "ibm-granite/granite-embedding-30m-english"
OUTPUT_DIR = "data/processed/"  # Directory to save the output file
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "embeddings_labeled.parquet")
BATCH_SIZE = 16  # Adjust based on your hardware


@st.cache_resource(show_spinner=False)
def load_model():
    """Load the tokenizer and model for embedding generation."""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model


def generate_embedding_batch(texts, tokenizer, model, device):
    """Generate embeddings for a batch of texts using the specified model."""
    inputs = tokenizer(texts, return_tensors="pt", truncation=True,
                       padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
    return embeddings


def app():
    st.title("Embedding Generator with Labels")

    uploaded_file = st.file_uploader("Select a CSV file", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("File loaded successfully!")
        st.dataframe(df.head())

        st.subheader("Columns for generating embedding text")
        embedding_columns = []
        for col in df.columns:
            if st.checkbox(f"Include column '{col}'", key=f"col_emb_{col}"):
                embedding_columns.append(col)

        id_column = st.selectbox("Select the ID column (optional)", [
                                 ""] + list(df.columns))
        label_column = st.selectbox("Select the label column (optional)", [
                                    ""] + list(df.columns))

        generate = False

        # Ensure the output directory exists
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)

        if os.path.exists(OUTPUT_FILE):
            st.warning(
                f"⚠️ The file '{OUTPUT_FILE}' already exists. Do you want to overwrite it?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Yes, overwrite file"):
                    generate = True
            with col2:
                if st.button("No, cancel operation"):
                    st.stop()
        else:
            if st.button("Generate embeddings and save file"):
                generate = True

        if generate:
            if not embedding_columns:
                st.warning(
                    "⚠️ Select at least one column to generate embeddings.")
                st.stop()

            # Initialize progress bar and status
            progress_bar = st.progress(0)
            status_text = st.empty()
            total_steps = 4  # Load model, prepare data, generate embeddings, save file
            current_step = 0

            # Step 1: Load model
            status_text.text("🔄 Loading model...")
            progress_bar.progress(current_step / total_steps)
            tokenizer, model = load_model()
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            current_step += 1
            progress_bar.progress(current_step / total_steps)

            # Step 2: Prepare data
            status_text.text("📝 Preparing data...")
            # Concatenate selected columns into a single text per row
            texts = df[embedding_columns].fillna(
                "").astype(str).agg(" ".join, axis=1)
            # Include label column in the text for embedding
            if label_column:
                labels = df[label_column].fillna("").astype(str)
                combined_texts = (texts + " " + labels).tolist()
            else:
                combined_texts = texts.tolist()
                labels = pd.Series(["" for _ in range(len(df))])
            ids = df[id_column].tolist() if id_column else list(range(len(df)))
            current_step += 1
            progress_bar.progress(current_step / total_steps)

            # Step 3: Generate embeddings
            status_text.text("🧠 Generating embeddings...")
            data = []
            num_batches = (len(combined_texts) + BATCH_SIZE - 1) // BATCH_SIZE
            with st.status("Processing batches...", expanded=True) as status:
                for i in tqdm(range(0, len(combined_texts), BATCH_SIZE), desc="Generating embeddings", total=num_batches):
                    batch_texts = combined_texts[i:i + BATCH_SIZE]
                    batch_ids = ids[i:i + BATCH_SIZE]
                    batch_labels = labels[i:i + BATCH_SIZE]

                    # Generate embeddings for the batch of combined texts
                    embeddings = generate_embedding_batch(
                        batch_texts, tokenizer, model, device)

                    for id_doc, emb, text, label in zip(batch_ids, embeddings, batch_texts, batch_labels):
                        data.append({
                            "id_doc": id_doc,
                            "embedding": emb.astype(np.float32),
                            "combined_text": text,
                            "label": str(label)  # Ensure label is a string
                        })

                    # Update progress within embedding step
                    batch_progress = (i + BATCH_SIZE) / len(combined_texts)
                    step_progress = (
                        current_step + batch_progress) / total_steps
                    progress_bar.progress(
                        min(step_progress, (current_step + 1) / total_steps))
                    status.update(
                        label=f"Processing batch {i // BATCH_SIZE + 1}/{num_batches}")

            current_step += 1
            progress_bar.progress(current_step / total_steps)

            # Step 4: Save file
            status_text.text("💾 Saving file...")
            result_df = pd.DataFrame(data)
            # Ensure label column is string type
            result_df['label'] = result_df['label'].astype(str)
            result_df.to_parquet(OUTPUT_FILE, index=False)
            current_step += 1
            progress_bar.progress(current_step / total_steps)

            status_text.text(f"✅ File saved successfully as: `{OUTPUT_FILE}`")
            st.write(result_df.head())


if __name__ == "__main__":
    app()

import streamlit as st
import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from tqdm import tqdm
import csv
import io

MODEL_NAME = "ibm-granite/granite-embedding-30m-english"
OUTPUT_DIR = "data/processed"  # Directory to save the output file
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


def detect_delimiter(file):
    """Detect the delimiter of a CSV or TXT file using csv.Sniffer."""
    file.seek(0)
    sample = file.read(1024).decode("utf-8")
    sniffer = csv.Sniffer()
    delimiter = sniffer.sniff(sample).delimiter
    file.seek(0)
    return delimiter


def preprocess_tab_delimited_file(file):
    """Preprocess a tab-delimited file to ensure each row has exactly 2 fields."""
    file.seek(0)
    lines = file.read().decode("utf-8").splitlines()
    cleaned_lines = []
    problematic_rows = []

    for i, line in enumerate(lines, 1):
        # Skip empty lines
        if not line.strip():
            continue
        # Split on tabs and strip any trailing tabs
        fields = line.rstrip("\t").split("\t")
        if len(fields) != 2:
            problematic_rows.append((i, line))
            # Join all fields after the first one as the message text
            fields = [fields[0], "\t".join(fields[1:])]
        cleaned_lines.append("\t".join(fields))

    # Report problematic rows
    if problematic_rows:
        st.warning(
            "Found rows with unexpected number of fields (should be 2). These have been fixed:")
        for row_num, row in problematic_rows:
            st.write(f"Line {row_num}: {row}")

    # Create a new file-like object with the cleaned data
    cleaned_data = "\n".join(cleaned_lines)
    return io.StringIO(cleaned_data)


def app():
    st.title("Embedding Generator with Labels")

    uploaded_file = st.file_uploader(
        "Select a CSV, TXT, or XLSX file", type=["csv", "txt", "xlsx"])

    if uploaded_file:
        file_extension = os.path.splitext(uploaded_file.name)[1].lower()

        delimiter = None
        if file_extension in [".csv", ".txt"]:
            delimiter_choice = st.selectbox(
                "Select the delimiter (or auto-detect)",
                ["Auto-detect", ", (comma)", "\t (tab)", "; (semicolon)"],
                format_func=lambda x: x if x == "Auto-detect" else x.split()[
                    0],
                index=2 if file_extension == ".txt" else 0
            )

            if delimiter_choice == "Auto-detect":
                try:
                    delimiter = detect_delimiter(uploaded_file)
                    st.info(f"Detected delimiter: '{delimiter}'")
                except Exception as e:
                    st.error(
                        f"Could not detect delimiter: {str(e)}. Please select a delimiter manually.")
                    st.stop()
            else:
                delimiter = delimiter_choice.split()[0]

        try:
            if file_extension in [".csv", ".txt"]:
                # Preprocess the file to handle extra tabs
                cleaned_file = preprocess_tab_delimited_file(uploaded_file)
                df = pd.read_csv(cleaned_file, sep="\t",
                                 names=["label", "text"])
            elif file_extension == ".xlsx":
                df = pd.read_excel(uploaded_file)
            else:
                st.error(
                    "Unsupported file type. Please upload a CSV, TXT, or XLSX file.")
                st.stop()
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()

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

            progress_bar = st.progress(0)
            status_text = st.empty()
            total_steps = 4
            current_step = 0

            status_text.text("🔄 Loading model...")
            progress_bar.progress(current_step / total_steps)
            tokenizer, model = load_model()
            device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            current_step += 1
            progress_bar.progress(current_step / total_steps)

            status_text.text("📝 Preparing data...")
            texts = df[embedding_columns].fillna(
                "").astype(str).agg(" ".join, axis=1)
            if label_column:
                labels = df[label_column].fillna("").astype(str)
                combined_texts = (texts + " " + labels).tolist()
            else:
                combined_texts = texts.tolist()
                labels = pd.Series(["" for _ in range(len(df))])
            ids = df[id_column].tolist() if id_column else list(range(len(df)))
            current_step += 1
            progress_bar.progress(current_step / total_steps)

            status_text.text("🧠 Generating embeddings...")
            data = []
            num_batches = (len(combined_texts) + BATCH_SIZE - 1) // BATCH_SIZE
            with st.status("Processing batches...", expanded=True) as status:
                for i in tqdm(range(0, len(combined_texts), BATCH_SIZE), desc="Generating embeddings", total=num_batches):
                    batch_texts = combined_texts[i:i + BATCH_SIZE]
                    batch_ids = ids[i:i + BATCH_SIZE]
                    batch_labels = labels[i:i + BATCH_SIZE]

                    embeddings = generate_embedding_batch(
                        batch_texts, tokenizer, model, device)

                    for id_doc, emb, text, label in zip(batch_ids, embeddings, batch_texts, batch_labels):
                        data.append({
                            "id_doc": id_doc,
                            "embedding": emb.astype(np.float32),
                            "combined_text": text,
                            "label": str(label)
                        })

                    batch_progress = (i + BATCH_SIZE) / len(combined_texts)
                    step_progress = (
                        current_step + batch_progress) / total_steps
                    progress_bar.progress(
                        min(step_progress, (current_step + 1) / total_steps))
                    status.update(
                        label=f"Processing batch {i // BATCH_SIZE + 1}/{num_batches}")

            current_step += 1
            progress_bar.progress(current_step / total_steps)

            status_text.text("💾 Saving file...")
            result_df = pd.DataFrame(data)
            result_df['label'] = result_df['label'].astype(str)
            result_df.to_parquet(OUTPUT_FILE, index=False)
            current_step += 1
            progress_bar.progress(current_step / total_steps)

            status_text.text(f"✅ File saved successfully as: `{OUTPUT_FILE}`")
            st.write(result_df.head())


if __name__ == "__main__":
    app()

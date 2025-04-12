import streamlit as st
import pandas as pd
import os
import json
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from tqdm import tqdm

MODEL_NAME = "ibm-granite/granite-embedding-30m-english"
OUTPUT_FILE = "embeddings_labeled.parquet"

@st.cache_resource(show_spinner=False)
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()
    return tokenizer, model

def gerar_embedding(texto, tokenizer, model, device):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
    return embedding

def app():
    st.title("Gerador de Embeddings com Label")

    uploaded_file = st.file_uploader("Selecione um arquivo CSV", type="csv")

    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("Arquivo carregado com sucesso!")
        st.dataframe(df.head())

        st.subheader("Colunas para gerar o texto de embeddings")
        embedding_columns = []
        for col in df.columns:
            if st.checkbox(f"Incluir coluna '{col}'", key=f"col_emb_{col}"):
                embedding_columns.append(col)

        id_column = st.selectbox("Selecione a coluna de ID (opcional)", [""] + list(df.columns))
        label_column = st.selectbox("Selecione a coluna de label (opcional)", [""] + list(df.columns))

        gerar = False

        if os.path.exists(OUTPUT_FILE):
            st.warning(f"⚠️ O arquivo '{OUTPUT_FILE}' já existe. Deseja sobrescrevê-lo?")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Sim, sobrescrever arquivo"):
                    gerar = True
            with col2:
                if st.button("Não, cancelar operação"):
                    st.stop()
        else:
            if st.button("Gerar embeddings e salvar arquivo"):
                gerar = True

        if gerar:
            if not embedding_columns:
                st.warning("⚠️ Selecione pelo menos uma coluna para gerar os embeddings.")
                st.stop()

            st.info("🔄 Carregando modelo e gerando embeddings...")
            tokenizer, model = load_model()
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)

            textos = df[embedding_columns].fillna("").astype(str).agg(" ".join, axis=1)
            ids = df[id_column] if id_column else list(range(len(df)))
            labels = df[label_column] if label_column else ["" for _ in range(len(df))]

            data = []
            for i, (id_doc, texto, label) in enumerate(tqdm(zip(ids, textos, labels), total=len(df))):
                emb = gerar_embedding(texto, tokenizer, model, device)
                data.append({
                    "id_doc": id_doc,
                    "embedding": emb.astype(np.float32),
                    "texto_agrupado": texto,
                    "label": label if pd.notnull(label) else ""
                })

            result_df = pd.DataFrame(data)
            result_df.to_parquet(OUTPUT_FILE, index=False)

            st.success(f"✅ Arquivo salvo com sucesso como: `{OUTPUT_FILE}`")
            st.write(result_df.head())

if __name__ == "__main__":
    app()
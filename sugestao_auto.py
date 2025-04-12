import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from utils import calculate_suggested_label
import os

PARQUET_FILE = "embeddings_labeled.parquet"

def app():
    st.title('Sugestão de Categoria')
    st.write("Sugere categorias baseado na similaridade.")

    st.sidebar.title("Configurações")
    n_similares = st.sidebar.slider("Número de Elementos Similares", 1, 25, 11)
    similarity_threshold = st.sidebar.slider("Threshold de Similaridade", 0.9, 1.0, 0.99)
    perc_filtro = st.sidebar.slider("Percentual da base a ser avaliada", 10, 100, 30)
    active_state = st.sidebar.checkbox("Classificação Automática", value=True)
    label_search_button = st.sidebar.button("Pesquisar por Rótulos")

    if label_search_button:
        if not os.path.exists(PARQUET_FILE):
            st.error(f"Arquivo '{PARQUET_FILE}' não encontrado.")
            return

        df = pd.read_parquet(PARQUET_FILE)
        if "label" not in df.columns:
            df["label"] = ""
        df = calculate_suggested_label(df, n_similares, similarity_threshold, perc_filtro)
        st.session_state.df = df

    if "df" in st.session_state:
        df = st.session_state.df
        distinct_labels = df["rotulo_sugerido"].dropna().unique()

        for label in distinct_labels:
            df_label = df[df["rotulo_sugerido"] == label]

            if active_state:
                df_label = df_label[df_label["media"] > similarity_threshold]

            original_count = len(df_label)
            df.loc[df_label.index, "label"] = label
            new_label_count = len(df_label[df_label["label"] == label])

            st.write(f"Rótulo: {label} - {new_label_count} instâncias ganharam um novo rótulo (de {original_count} instâncias possíveis).")

        try:
            df.to_parquet(PARQUET_FILE, index=False)
            st.success("Todos os rótulos foram atualizados e salvos com sucesso no arquivo.")
        except Exception as e:
            st.error(f"Erro ao salvar o arquivo: {e}")

if __name__ == "__main__":
    app()

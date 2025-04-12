import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
from src.utilities.helpers import calculate_suggested_label
import os

PARQUET_FILE = "embeddings_labeled.parquet"

def app():
    st.title('Sugestão de Categoria')
    st.write("Sugere categorias baseado na similaridade.")

    st.sidebar.title("Configurações")
    n_similares = st.sidebar.slider("Número de Elementos Similares", 1, 25, 5)
    similarity_threshold = st.sidebar.slider("Threshold de Similaridade", 0.8, 1.0, 0.99)
    n_view = st.sidebar.slider("Número de Elementos Display", 1, 30, 25)
    perc_filtro = st.sidebar.slider("Percentual da base a ser avaliada", 10, 100, 30)
    label_search_button = st.sidebar.button("Gerar Sugestões")

    if label_search_button:
        if not os.path.exists(PARQUET_FILE):
            st.error(f"Arquivo '{PARQUET_FILE}' não encontrado. Gere os embeddings primeiro.")
            return

        df = pd.read_parquet(PARQUET_FILE)
        if 'label' not in df.columns:
            df['label'] = ""

        df = calculate_suggested_label(df, n_similares, similarity_threshold, perc_filtro)
        st.session_state.df = df
        st.session_state.current_label_index = 0

    if 'df' in st.session_state:
        df = st.session_state.df
        distinct_labels = df['rotulo_sugerido'].dropna().unique()

        label_index = st.session_state.get('current_label_index', 0)
        next_back_cols = st.columns([1, 1, 2])
        with next_back_cols[0]:
            if st.button("BACK"):
                st.session_state.current_label_index = (label_index - 1) % len(distinct_labels)
                label_index = st.session_state.current_label_index
        with next_back_cols[1]:
            if st.button("NEXT"):
                st.session_state.current_label_index = (label_index + 1) % len(distinct_labels)
                label_index = st.session_state.current_label_index

        if label_index < len(distinct_labels):
            current_label = distinct_labels[label_index]
            current_elements = df[df['rotulo_sugerido'] == current_label].head(n_view)

            st.header(f"Rótulo Atual: {current_label}")

            selected_similars = []
            for idx, row in current_elements.iterrows():
                col1, col2 = st.columns([7, 1])
                with col1:
                    if st.checkbox(f"{row['texto_agrupado'][0:600]}", key=idx, value=True):
                        selected_similars.append(idx)
                with col2:
                    st.write(f"Label: {row['label']}")

            if st.button("ROTULAR"):
                for idx in selected_similars:
                    df.at[idx, 'label'] = current_label

                st.session_state.df = df
                try:
                    df.to_parquet(PARQUET_FILE, index=False)
                    st.success("Elementos rotulados com sucesso e arquivo atualizado!")
                except Exception as e:
                    st.error(f"Erro ao salvar arquivo: {e}")
        else:
            st.info("Todas as rotulações foram confirmadas.")

if __name__ == "__main__":
    app()

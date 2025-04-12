import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from src.utilities.helpers import calculate_similarity
import os

PARQUET_FILE = "embeddings_labeled.parquet"


def app():
    lista_tipo = ["", "SENSIVEL", "NAO SENSIVEL"]
    st.title('Cluster')
    st.write("Sugere clusters para facilitar classificação.")

    st.sidebar.title("Configurações")
    n_similares = st.sidebar.slider("Número de Elementos Similares", 1, 30, 10)
    n_clusters = st.sidebar.slider("Número de Clusters", 2, 50, 5)
    search_words = st.sidebar.text_input("Pesquisar Palavras", "")
    active_state = st.sidebar.checkbox("Pesquisar em rotulados", value=False)
    selected_tipo = st.selectbox("Selecione um tipo", lista_tipo)
    submit_button = st.sidebar.button("Submeter")

    if submit_button:
        if not os.path.exists(PARQUET_FILE):
            st.error(f"Arquivo '{PARQUET_FILE}' não encontrado.")
            return

        df = pd.read_parquet(PARQUET_FILE)
        if 'label' not in df.columns:
            df['label'] = ""

        if selected_tipo:
            df = df[df['label'] == selected_tipo]
        elif search_words:
            df = df[df['texto_agrupado'].str.contains(search_words, case=False, na=False)]

        if df.shape[0] > 3:
            
            df = calculate_similarity(df, n_clusters)
        else:
            
            df['cluster'] = 0

        st.session_state.df = df
        st.session_state.cluster = 0
        st.session_state.active_state = active_state

    if 'df' in st.session_state:
        df = st.session_state.df
        distinct_labels = df['label'].unique().tolist()
        distinct_labels = [x for x in distinct_labels if x != ""]
        distinct_labels.sort()

        if not st.session_state.active_state:
            df = df[df['label'] == ""]

        cluster = st.session_state.cluster

        st.header("Parâmetros")
        st.write(f"Exibindo {n_similares} elementos similares")
        st.write(f"Cluster: {cluster+1} de {n_clusters}")
        labels_count = df.shape[0]
        empty_labels_count = df['label'].eq("").sum()
        st.write(f"Número de elementos sem rótulo: {empty_labels_count}/{labels_count}")

        next_back_cols = st.columns([1, 1, 2])
        with next_back_cols[0]:
            if st.button("BACK"):
                st.session_state.cluster = (cluster - 1) % n_clusters
                cluster = st.session_state.cluster
        with next_back_cols[1]:
            if st.button("NEXT"):
                st.session_state.cluster = (cluster + 1) % n_clusters
                cluster = st.session_state.cluster

        if not st.session_state.active_state:
            similar_docs = df[(df['cluster'] == cluster) & (df['label'] == "")]
        else:
            similar_docs = df[df['cluster'] == cluster]
        similar_docs = similar_docs.head(n_similares)

        st.header("Seleção de Rótulos dos Clusters")

        #if 'rótulos_existentes' not in st.session_state:
        #    st.session_state.rótulos_existentes = ["", "SENSIVEL", "NAO SENSIVEL"]
        #    st.session_state.rótulos_existentes.sort()

        label_cols = st.columns([2, 3, 2])
        with label_cols[0]:
            selected_label = st.selectbox("Selecione um rótulo", st.session_state.rótulos_existentes)
        with label_cols[1]:
            new_label = st.text_input("Adicionar Novo Rótulo")
        with label_cols[2]:
            if st.button("Adicionar Rótulo"):
                if new_label and new_label not in st.session_state.rótulos_existentes:
                    st.session_state.rótulos_existentes.append(new_label)
                    selected_label = new_label
                    st.success(f"Rótulo '{new_label}' adicionado com sucesso.")

        st.header("Elementos Similares do Cluster")

        selected_similars = []
        for idx, row in similar_docs.iterrows():
            col1, col2 = st.columns([7, 1])
            with col1:
                if st.checkbox(f"{row['texto_agrupado'][0:2500]}", key=idx, value=True):
                    selected_similars.append(idx)
            with col2:
                st.write(f"Label: {row['label']}")

        if st.button("ROTULAR"):
            df_original = st.session_state.df
            for idx in selected_similars:
                df_original.at[idx, 'label'] = selected_label
            st.session_state.df = df_original
            try:
                df_original.to_parquet(PARQUET_FILE, index=False)
                st.success("Elementos rotulados com sucesso e arquivo atualizado!")
            except Exception as e:
                st.error(f"Erro ao salvar arquivo: {e}")

if __name__ == "__main__":
    app()
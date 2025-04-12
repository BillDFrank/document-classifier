import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
from src.utilities.helpers import persist_labels, calculate_similarity, propose_cluster_names
import os

PARQUET_FILE = "embeddings_labeled.parquet"

def perform_clustering(df, n_clusters):
    """Performs K-Means clustering on the dataset."""
    if df.empty:
        return None, None

    # Feature Engineering
    enc = OneHotEncoder(handle_unknown="ignore")
    X = enc.fit_transform(df[['texto_agrupado']]).toarray()

    # Apply K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X)

    return df, kmeans

def app():
    st.title("Cluster")
    st.write("Sugere clusters para facilitar classificação.")

    st.sidebar.title("Configurações")
    n_similares = st.sidebar.slider("Número de Elementos Similares", 1, 30, 10)
    n_clusters = st.sidebar.slider("Número de Clusters", 2, 50, 10)
    submit_button = st.sidebar.button("Submeter")

    # Load DataFrame from Parquet file
    if "df" not in st.session_state:
        if not os.path.exists(PARQUET_FILE):
            st.error(f"Arquivo '{PARQUET_FILE}' não encontrado. Gere os embeddings primeiro.")
            return
        else:
            st.session_state.df = pd.read_parquet(PARQUET_FILE)

    # Perform clustering only when "Submeter" is clicked
    if submit_button:
        df = st.session_state.df
        df, kmeans = perform_clustering(df, n_clusters)

        if df is not None:
            df = calculate_similarity(df, n_clusters)
            st.session_state.df = df
            st.session_state.cluster = 0
            st.success("Clusterização concluída com sucesso!")

    # If clustering has been performed, show the labeling interface
    if "cluster" in st.session_state:
        df = st.session_state.df
        distinct_labels = df['label'].unique().tolist()
        distinct_labels = [x for x in distinct_labels if x is not None]
        distinct_labels.sort()
        df['label'] = df['label'].fillna("")
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

        similar_docs = df[(df['cluster'] == cluster) & (df['label'] == "")]
        similar_docs = similar_docs.head(n_similares)

        st.header("Seleção de Rótulos dos Clusters")

        if 'rótulos_existentes' not in st.session_state:
            if len(distinct_labels) < 2:
                st.session_state.rótulos_existentes = [
                    "SIGILOSO", "NAO SIGILOSO"
                ]
                st.session_state.rótulos_existentes.sort()
            else:
                st.session_state.rótulos_existentes = distinct_labels

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
            if st.checkbox(f"{row['texto_agrupado'][0:1000]}", key=idx, value=True):
                selected_similars.append(idx)

        if st.button("ROTULAR"):
            for idx in selected_similars:
                st.session_state.df.at[idx, 'label'] = selected_label

            try:
                st.session_state.df.to_parquet(PARQUET_FILE, index=False)
                st.success("Elementos rotulados com sucesso e arquivo atualizado!")
            except Exception as e:
                st.error(f"Erro ao salvar arquivo: {e}")

if __name__ == "__main__":
    app()

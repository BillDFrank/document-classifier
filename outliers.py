import networkx as nx
import community
from itertools import combinations
from torch.utils.data import DataLoader
import streamlit as st
import pandas as pd
import numpy as np
from opensearchpy import OpenSearch, RequestsHttpConnection
from utils import connect_opensearch, fetch_documents, create_dataframe, persist_labels


def calculate_similarity_split(df, threshold, selected_label_):
    # Filtrar o DataFrame com base no label selecionado
    df_rotulados = df[df['label'] == selected_label_].copy()
    
    # Obter os embeddings completos
    embeddings = np.array(df_rotulados["embedding_completo"].tolist())

    # Construir o grafo
    G = nx.Graph()
    
    # Adicionar nós ao grafo
    G.add_nodes_from(df_rotulados.index)
    
    # Combinações de pares para calcular similaridade
    pairs = list(combinations(df_rotulados.index, 2))
    
    # Calcular similaridade utilizando embeddings
    similarity_scores = []
    for idx1, idx2 in pairs:
        similarity = np.dot(embeddings[df_rotulados.index.get_loc(idx1)], embeddings[df_rotulados.index.get_loc(idx2)]) / (
            np.linalg.norm(embeddings[df_rotulados.index.get_loc(idx1)]) * np.linalg.norm(embeddings[df_rotulados.index.get_loc(idx2)])
        )
        similarity_scores.append(similarity)
    
    # Adicionar arestas ao grafo com base no threshold
    for i, (idx1, idx2) in enumerate(pairs):
        if similarity_scores[i] > threshold:
            G.add_edge(idx1, idx2)
    
    # Aplicar o algoritmo de Louvain para detecção de comunidades
    partition = community.best_partition(G)
    
    # Associar os rótulos de cluster ao DataFrame
    df_rotulados['cluster'] = df_rotulados.index.map(partition)
    
    # Identificar outliers e atribuí-los ao cluster -1
    for idx in df_rotulados.index:
        if G.degree[idx] == 0:  # Sem conexões, é outlier
            df_rotulados.at[idx, 'cluster'] = -1
    
    return df_rotulados


def app():
    st.title('Cluster')
    st.write("Identificar clusters e outliers.")

    # Sidebar configuration
    st.sidebar.title("Configurações")
    
    # Conectar ao OpenSearch e buscar documentos
    if 'es' not in st.session_state:
        HOST = "10.10.25.161"
        PORT = 9200
        es = connect_opensearch(HOST, PORT)
        st.session_state.es = es
    else:
        es = st.session_state.es

    opensearch_index = "classificador_dados_sensiveis"

    
    if not es.ping():
        st.sidebar.error("Não foi possível conectar ao OpenSearch")
    else:
        st.sidebar.success("Conexão bem-sucedida")
    
    if 'documents' not in st.session_state:
        documents = fetch_documents(es, opensearch_index)
        st.session_state.documents = documents
    else:
        documents = st.session_state.documents
    
    if documents:
        df = create_dataframe(documents)
        distinct_labels = df['label'].unique().tolist()
        distinct_labels.sort()
        
        if len(distinct_labels) == 0:
            st.sidebar.error("Deve haver rótulos já estabelecidos para permitir divisão.")
        else:
            if 'rótulos_existentes' not in st.session_state:
                st.session_state.rótulos_existentes = distinct_labels
    
    threshold = st.sidebar.slider("Threshold de Similaridade", 0.5, 1.0, 0.99)
    selected_label_ = st.sidebar.selectbox("Selecione um rótulo", st.session_state.rótulos_existentes, key="selected_label_")
    submit_button = st.sidebar.button("Submeter")

    if submit_button:
        df = calculate_similarity_split(df, threshold, selected_label_)
        st.session_state.df = df
        st.session_state.cluster = -1  # Começar exibindo os outliers
        st.session_state.n_clusters = df['cluster'].nunique()

    if 'df' in st.session_state:
        df = st.session_state.df
        cluster = st.session_state.cluster
        n_clusters = st.session_state.n_clusters

        st.header("Parâmetros")
        labels_count = df.shape[0]
        cluster_count = df['cluster'].eq(cluster).sum()
        st.write(f"Número de elementos no cluster: {cluster_count}/{labels_count}")
        
        next_back_cols = st.columns([1, 1, 2])
        with next_back_cols[0]:
            if st.button("BACK"):
                st.session_state.cluster = (cluster - 1) % n_clusters
                cluster = st.session_state.cluster
        with next_back_cols[1]:
            if st.button("NEXT"):
                st.session_state.cluster = (cluster + 1) % n_clusters
                cluster = st.session_state.cluster

        # Atualizar a exibição dos documentos conforme o cluster atual
        similar_docs = df[df['cluster'] == cluster]

        st.header("Seleção de Rótulos dos Clusters")

        label_cols = st.columns([2, 3, 2])
        with label_cols[0]:
            selected_label = st.selectbox("Selecione um rótulo", st.session_state.rótulos_existentes, key=f"selected_label_{cluster}")
        with label_cols[1]:
            new_label = st.text_input("Adicionar Novo Rótulo")
        with label_cols[2]:
            if st.button("Adicionar Rótulo"):
                if new_label and new_label not in st.session_state.rótulos_existentes:
                    st.session_state.rótulos_existentes.append(new_label)
                    st.session_state.rótulos_existentes.sort()
                    selected_label = new_label
                    st.success(f"Rótulo '{new_label}' adicionado com sucesso.")
        
        st.header("Elementos Similares do Cluster")

        selected_similars = []
        for idx, row in similar_docs.iterrows():
            col1, col2 = st.columns([7, 1])
            with col1:
                if st.checkbox(f"{row['ds_documento_ocr'][0:1200]}", key=f"chk_{idx}_{cluster}", value=True):
                    selected_similars.append(idx)
            with col2:
                st.write(f"Label: {row['label']}")

        if st.button("ROTULAR"):
            for idx in selected_similars:
                df.at[idx, 'label'] = selected_label
            atualizou = persist_labels(es, opensearch_index, df)
            if atualizou:
                st.success("Elementos rotulados com sucesso!")
            else:
                st.error("Falha ao rotular elementos!")

if __name__ == "__main__":
    app()


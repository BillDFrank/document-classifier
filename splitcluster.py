import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation as LDA
from opensearchpy import OpenSearch, RequestsHttpConnection
from utils import connect_opensearch, fetch_documents, create_dataframe, persist_labels, propose_cluster_names, calculate_suggested_label

# Funções auxiliares
def calculate_similarity_split(df, n_clusters, selected_label_):
    df_rotulados = df[df['label'] == selected_label_].copy()
    embeddings = np.array(df_rotulados["embedding_completo"].tolist())
    model = KMeans(n_clusters=n_clusters, n_init=10)
    labels = model.fit_predict(embeddings)
    df_rotulados.loc[:, 'cluster'] = labels
    return df_rotulados

def app():
    st.title('Cluster')
    st.write("Dividir clusters.")

    # Layout do aplicativo
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

    # Initialize cluster and DataFrame in session state
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'cluster' not in st.session_state:
        st.session_state.cluster = 0

    n_similares = st.sidebar.slider("Número de Elementos Similares", 1, 25, 10)
    n_clusters = st.sidebar.slider("Número de Clusters", 2, 20, 2)
    selected_label_ = st.sidebar.selectbox("Selecione um rótulo", st.session_state.rótulos_existentes, key="selected_label_")
    
    submit_button = st.sidebar.button("Submeter")

    if submit_button:
        df = calculate_similarity_split(df, n_clusters, selected_label_)
        st.session_state.df = df
        st.session_state.cluster = 0

    if st.session_state.df is not None:
        df = st.session_state.df
        cluster = st.session_state.cluster
        st.header("Parâmetros")
        st.write(f"Exibindo {n_similares} elementos similares")
        st.write(f"Cluster: {cluster+1} de {n_clusters}")
        labels_count = df.shape[0]
        cluster_count = df['cluster'].eq(cluster).sum()
        st.write(f"Número de elementos sem rótulo: {cluster_count}/{labels_count}")
        
        next_back_cols = st.columns([1, 1, 2])
        with next_back_cols[0]:
            if st.button("BACK"):
                st.session_state.cluster = (cluster - 1) % n_clusters
                cluster = st.session_state.cluster
        with next_back_cols[1]:
            if st.button("NEXT"):
                st.session_state.cluster = (cluster + 1) % n_clusters
                cluster = st.session_state.cluster

        similar_docs = df[(df['cluster'] == cluster)]
        similar_docs = similar_docs.head(n_similares)
        
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
                st.write(f"Label: ")

        # Use session state to handle the apply_to_all state
        if 'apply_to_all' not in st.session_state:
            st.session_state.apply_to_all = False

        st.session_state.apply_to_all = st.checkbox(
            "Aplicar novo rótulo para todas as instâncias no cluster atual", 
            value=st.session_state.apply_to_all
        )

        if st.button("ROTULAR"):
            # Determine the number of affected instances
            if st.session_state.apply_to_all:
                affected_instances = df[df['cluster'] == cluster].shape[0]
            else:
                affected_instances = len(selected_similars)

            # Set a flag in session state to wait for confirmation
            st.session_state.confirm_labeling = True
            st.session_state.affected_instances = affected_instances

        # Show the confirm button only if ROTULAR was clicked
        if st.session_state.get('confirm_labeling', False):
            if st.button(f"Confirmar Rotulação para {st.session_state.affected_instances} instâncias"):
                if st.session_state.apply_to_all:
                    # Apply label to all instances in the current cluster
                    df.loc[df['cluster'] == cluster, 'label'] = selected_label
                else:
                    # Apply label only to selected instances
                    for idx in selected_similars:
                        df.at[idx, 'label'] = selected_label

                # Persist labels after confirmation
                atualizou = persist_labels(es, opensearch_index, df)
                if atualizou:
                    st.success("Elementos rotulados com sucesso!")
                else:
                    st.error("Falha ao rotular elementos!")

                # Reset confirmation state
                st.session_state.confirm_labeling = False

if __name__ == "__main__":
    app()














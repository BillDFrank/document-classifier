import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
import streamlit as st
from opensearchpy import OpenSearch
from utils import connect_opensearch, fetch_documents, create_dataframe, persist_labels
import community

# Função para identificar similares e formar clusters

def calculate_similarity_split(df, threshold, selected_label_):
    """
    Identifica clusters de documentos similares com base no threshold de similaridade e retorna um DataFrame com os clusters.

    :param df: DataFrame contendo os documentos
    :param threshold: Limite de similaridade para definir conexões no grafo
    :param selected_label_: Label utilizado para filtrar os documentos
    :return: DataFrame com cluster atribuído a cada documento
    """
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
        if idx1 != idx2:  
            similarity = np.dot(embeddings[df_rotulados.index.get_loc(idx1)], embeddings[df_rotulados.index.get_loc(idx2)]) / (
                np.linalg.norm(embeddings[df_rotulados.index.get_loc(idx1)]) * np.linalg.norm(embeddings[df_rotulados.index.get_loc(idx2)])
            )
            similarity_scores.append(similarity)

    # Adicionar arestas ao grafo com base no threshold
    # Lista para armazenar similaridades acima do threshold
    similarities_above_threshold = []

    # Adicionar arestas ao grafo com base no threshold e armazenar as similaridades
    for i, (idx1, idx2) in enumerate(pairs):
        if similarity_scores[i] > threshold:
            G.add_edge(idx1, idx2)
            similarities_above_threshold.append(similarity_scores[i])

    # Calcular a média das similaridades acima do threshold
    if similarities_above_threshold:
        average_similarity = np.mean(similarities_above_threshold)
        st.write(f"**Valor médio das similaridades acima do threshold ({threshold}):** {average_similarity:.4f}")
    else:
        st.write(f"**Nenhuma similaridade acima do threshold ({threshold}).**")


    # Aplicar o algoritmo de Louvain para detecção de comunidades
    partition = community.best_partition(G)

    # Associar os rótulos de cluster ao DataFrame
    df_rotulados['cluster'] = df_rotulados.index.map(lambda idx: partition.get(idx, -1))
    cluster_counts = df_rotulados['cluster'].value_counts()

    # Identificar clusters com um único elemento
    clusters_with_one_element = cluster_counts[cluster_counts == 1].index

    # Atualizar o cluster desses elementos para 0
    df_rotulados.loc[df_rotulados['cluster'].isin(clusters_with_one_element), 'cluster'] = 0
    
    # Ajustar os valores dos clusters para serem sequenciais
    unique_clusters = sorted(df_rotulados['cluster'].unique())  # Obter valores únicos ordenados
    cluster_mapping = {old: new for new, old in enumerate(unique_clusters)}  # Criar mapeamento sequencial
    df_rotulados['cluster'] = df_rotulados['cluster'].map(cluster_mapping)  # Aplicar mapeamento ao DataFrame
    
    st.write(f"**Lista de clusters ajustados ({df_rotulados['cluster'].tolist()}).**")

    return df_rotulados

# Streamlit app para exibir e gerenciar clusters
def app():
    st.title('Clusters de Documentos Similares')
    st.write("Identificar e navegar entre clusters de documentos similares.")

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
        selected_label = st.sidebar.selectbox("Selecione um rótulo", distinct_labels)

        threshold = st.sidebar.slider("Threshold de Similaridade", 0.999, 1.0, 0.9999, step=0.0001)
        submit_button = st.sidebar.button("Submeter")

        if submit_button:
            df = calculate_similarity_split(df, threshold, selected_label)
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
            selected_label = st.selectbox("Selecione um rótulo", st.session_state.df['label'].unique(), key=f"selected_label_{cluster}")
        with label_cols[1]:
            new_label = st.text_input("Adicionar Novo Rótulo")
        with label_cols[2]:
            if st.button("Adicionar Rótulo"):
                if new_label and new_label not in st.session_state.df['label'].unique():
                    st.session_state.df['label'] = st.session_state.df['label'].append(pd.Series([new_label])).unique()
                    st.success(f"Rótulo '{new_label}' adicionado com sucesso.")

        st.header("Elementos Similares do Cluster")

        selected_similars = []
        for idx, row in similar_docs.iterrows():
            col1, col2 = st.columns([7, 1])
            with col1:
                if st.checkbox(f"{row['ds_documento_ocr'][:1200]}", key=f"chk_{idx}_{cluster}", value=True):
                    selected_similars.append(idx)
            with col2:
                st.write(f"Label: {row['label']}")

        if st.button("DELETAR DO INDEX"):
            if selected_similars:
                for idx in selected_similars:
                    try:
                        # Procurar o id_documento
                        id_documento = df.loc[idx, 'id_documento']
                        st.write(f"Id_documento: {id_documento}")

                        # Buscar os documentos no OpenSearch com o mesmo id_documento
                        query = {
                            "query": {
                                "match": {
                                    "id_documento": id_documento
                                }
                            }
                        }
                        response = es.search(index=opensearch_index, body=query)

                        # Obter os IDs dos documentos no OpenSearch
                        docs_to_delete = [hit["_id"] for hit in response["hits"]["hits"]]
                        st.write(f"docs_to_delete: {docs_to_delete}")

                        # Garantir que pelo menos uma instância permaneça
                        #if docs_to_delete:
                        #    docs_to_delete = docs_to_delete[1:]  # Deixa o primeiro e remove os outros

                        # Remover os documentos restantes do OpenSearch
                        for doc_id in docs_to_delete:
                            es.delete(index=opensearch_index, id=doc_id)
                            st.success(f"Documento {doc_id} deletado com sucesso!")

                    except Exception as e:
                        st.error(f"Erro ao deletar documento {idx}: {e}")
            else:
                st.warning("Nenhum documento selecionado para exclusão.")

if __name__ == "__main__":
    app()

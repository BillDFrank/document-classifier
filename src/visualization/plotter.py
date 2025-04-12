import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.utilities.helpers import connect_opensearch, fetch_documents, create_dataframe, persist_labels

def calculate_clusters(df, n_clusters, selected_labels):
    # Filtrar os documentos pelos rótulos selecionados
    df_rotulados = df[df['label'].isin(selected_labels)].copy()
    embeddings = np.array(df_rotulados["embedding_completo"].tolist())
    
    # Aplicar KMeans para criar os clusters
    model = KMeans(n_clusters=n_clusters, n_init=10)
    labels = model.fit_predict(embeddings)
    
    df_rotulados['cluster'] = labels
    return df_rotulados, embeddings

def plot_clusters(df, embeddings, n_clusters):
    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    df['PCA1'] = reduced_embeddings[:, 0]
    df['PCA2'] = reduced_embeddings[:, 1]
    
    # Aumenta o tamanho do texto mostrado no hover
    df['ds_documento_ocr'] = df['ds_documento_ocr'].str[:200]  # Exibe até 2000 caracteres

    # Configuração de cores mais discrepantes para os clusters
    color_discrete_map = px.colors.qualitative.Safe  # Usar uma paleta de cores com maior contraste

    fig = px.scatter(
        df, x='PCA1', y='PCA2', color='cluster',
        color_discrete_sequence=color_discrete_map,
        hover_data={'PCA1': False, 'PCA2': False, 'ds_documento_ocr': True},
        labels={'ds_documento_ocr': 'Texto do Documento'}
    )
    
    fig.update_traces(marker=dict(size=8, opacity=0.7),
                      selector=dict(mode='markers'))
    fig.update_layout(title_text=f'Visualização dos Clusters (n={n_clusters})',
                      title_x=0.5,
                      xaxis_title='Componente Principal 1',
                      yaxis_title='Componente Principal 2')
    
    st.plotly_chart(fig)

def app():
    st.title('Avaliação de Clusters')
    st.write("Visualizar separação de clusters baseados em rótulos selecionados.")

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
            st.sidebar.error("Deve haver rótulos já estabelecidos para permitir a avaliação.")
        else:
            # Seletor para o tipo de visualização
            view_type = st.sidebar.radio("Tipo de Visualização", ["Único Rótulo", "Múltiplos Rótulos"])
            
            if view_type == "Único Rótulo":
                selected_label = st.sidebar.selectbox("Selecione um rótulo", distinct_labels)
                n_clusters = st.sidebar.slider("Número de Clusters", 2, 20, 5)
                submit_button = st.sidebar.button("Submeter")
                
                if submit_button:
                    df_clustered, embeddings = calculate_clusters(df, n_clusters, [selected_label])
                    st.session_state.df_clustered = df_clustered
                    st.session_state.embeddings = embeddings
                    st.session_state.n_clusters = n_clusters

            elif view_type == "Múltiplos Rótulos":
                selected_labels = st.sidebar.multiselect("Selecione rótulos", distinct_labels)
                n_clusters = st.sidebar.slider("Número de Clusters", 2, 20, 5)
                submit_button = st.sidebar.button("Submeter")
                
                if submit_button and selected_labels:
                    df_clustered, embeddings = calculate_clusters(df, n_clusters, selected_labels)
                    st.session_state.df_clustered = df_clustered
                    st.session_state.embeddings = embeddings
                    st.session_state.n_clusters = n_clusters

    if 'df_clustered' in st.session_state:
        if view_type == "Único Rótulo":
            st.write(f"Visualizando clusters para o rótulo: {selected_label} com {st.session_state.n_clusters} clusters.")
        elif view_type == "Múltiplos Rótulos":
            st.write(f"Visualizando clusters para os rótulos: {', '.join(selected_labels)} com {st.session_state.n_clusters} clusters.")
        
        plot_clusters(st.session_state.df_clustered, st.session_state.embeddings, st.session_state.n_clusters)

if __name__ == "__main__":
    app()




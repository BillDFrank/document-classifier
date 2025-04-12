import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from opensearchpy import OpenSearch
from src.utilities.helpers import connect_opensearch, fetch_documents, create_dataframe

def generate_label_sequence(group):
    """Generate a label sequence that includes label counts for each envelope."""
    label_counts = group['label'].value_counts().sort_index()
    label_sequence = '->'.join([f"{label}:{count}" for label, count in label_counts.items()])
    return label_sequence

def aggregate_embeddings(embeddings):
    """Aggregates embeddings by summing them (or use another aggregation technique)."""
    return np.sum(np.stack(embeddings), axis=0)

def app():
    st.title('Comparação de Processos')
    st.write("Identifica processos similares com base em um processo de referência.")

    # Entrada do envelope de referência
    reference_envelope = st.sidebar.text_input("Digite o número do envelope de referência")
    
    if reference_envelope:
        # Conectar ao OpenSearch e buscar documentos
        HOST = "10.10.25.161"
        PORT = 9200
        es = connect_opensearch(HOST, PORT)
        if not es.ping():
            st.sidebar.error("Não foi possível conectar ao OpenSearch")
            return
        else:
            st.sidebar.success("Conexão bem-sucedida")
        
        # Layout do aplicativo
        st.sidebar.title("Configurações")
        opensearch_index = st.sidebar.text_input("OpenSearch Index", "classificador_dados_sensiveis")
        documents = fetch_documents(es, opensearch_index)
        
        if documents:
            df = create_dataframe(documents)
            
            # Garantir que apenas labels válidos sejam usados
            df_filtered = df[df['label'].notna() & (df['label'] != "")]
            
            if df_filtered.empty:
                st.error("Nenhum documento com rótulos válidos foi encontrado.")
                return
            
            st.session_state.df = df_filtered
            
            # Debugging: Display a sample of the filtered dataframe
            #st.write("Dados filtrados (Amostra):")
            #st.write(df_filtered.head())

            # Agrupar documentos por envelope e criar uma sequência de labels com contagem
            df_grouped = df_filtered.groupby('envelope').apply(generate_label_sequence).reset_index()
            df_grouped.columns = ['envelope', 'label_sequence']

            # Debugging: Display grouped dataframe to check label sequences
            #st.write("Dados agrupados por envelope (Amostra):")
            #st.write(df_grouped.head())

            # Identificar a label_sequence do envelope de referência
            reference_label_sequence = df_grouped[df_grouped['envelope'] == reference_envelope]['label_sequence'].values

            if len(reference_label_sequence) == 0:
                st.warning("Envelope de referência não encontrado ou não possui labels válidos.")
                return
            
            reference_label_sequence = reference_label_sequence[0]
            st.write(f"Sequência de labels do envelope de referência: {reference_label_sequence}")
            
            # Identificar processos com a mesma sequência de labels
            common_envelopes = df_grouped[df_grouped['label_sequence'] == reference_label_sequence]['envelope']
            df_same_sequence = df_filtered[df_filtered['envelope'].isin(common_envelopes)]
            
            # Debugging: Display dataframe with the same sequence of labels
            #st.write("Documentos com a mesma sequência de labels:")
            #st.write(df_same_sequence.head())

            # Agregar embeddings por processo
            df_embeddings = df_same_sequence.groupby('envelope')['embedding_completo'].apply(lambda x: aggregate_embeddings(x.tolist())).reset_index()

            # Calcular a similaridade dos embeddings usando similaridade cosseno
            similarity_results = []
            reference_embedding = df_embeddings[df_embeddings['envelope'] == reference_envelope]['embedding_completo'].values

            if len(reference_embedding) == 0:
                st.error("Nenhum embedding encontrado para o envelope de referência.")
                return
            
            reference_embedding = reference_embedding[0]

            for i in range(len(df_embeddings)):
                envelope = df_embeddings.iloc[i]['envelope']
                if envelope == reference_envelope:
                    continue
                
                sim = cosine_similarity([reference_embedding], [df_embeddings.iloc[i]['embedding_completo']])[0][0]
                similarity_results.append({
                    'envelope': envelope,
                    'similarity': sim
                })

            # Check if similarity results are available
            if not similarity_results:
                st.warning("Nenhum processo similar encontrado.")
                return

            # Converter resultados para DataFrame e encontrar o processo mais similar
            df_similarity = pd.DataFrame(similarity_results)
            most_similar_process = df_similarity.loc[df_similarity['similarity'].idxmax()]
            
            st.write(f"Processo mais similar ao envelope {reference_envelope}: {most_similar_process['envelope']} com similaridade de {most_similar_process['similarity']:.4f}")

            # Exibe o conteúdo de 'ds_documento_ocr' lado a lado com os labels
            proc1_docs = df_filtered[df_filtered['envelope'] == reference_envelope].sort_values(by='label')[['label', 'ds_documento_ocr']].values.tolist()
            proc2_docs = df_filtered[df_filtered['envelope'] == most_similar_process['envelope']].sort_values(by='label')[['label', 'ds_documento_ocr']].values.tolist()
            
            # Garantir que os documentos sejam exibidos lado a lado, mesmo que os números de documentos sejam diferentes
            max_docs = max(len(proc1_docs), len(proc2_docs))
            proc1_docs.extend([["", ""]] * (max_docs - len(proc1_docs)))  # Preenche com strings vazias
            proc2_docs.extend([["", ""]] * (max_docs - len(proc2_docs)))

            for (label1, doc1), (label2, doc2) in zip(proc1_docs, proc2_docs):
                col1, col2 = st.columns(2)
                with col1:
                    st.text(f"Label: {label1}")
                    st.text_area("Processo de Referência", doc1[:1000], height=150)
                with col2:
                    st.text(f"Label: {label2}")
                    st.text_area("Processo Similar", doc2[:1000], height=150)

if __name__ == "__main__":
    app()

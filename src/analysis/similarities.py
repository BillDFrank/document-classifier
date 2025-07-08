import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def app():
    st.title('Clusters de Documentos Similares')
    st.write("Esta aplicação permite visualizar documentos similares com base em seus embeddings.")

    if 'df' not in st.session_state:
        st.warning("Nenhum dado carregado. Por favor, carregue um arquivo Parquet na seção 'Datasource'.")
        return

    df = st.session_state.df

    if df.empty:
        st.warning("O DataFrame está vazio. Por favor, carregue dados válidos na seção 'Datasource'.")
        return

    # Ensure 'label' column contains strings and filter invalid entries
    df['label'] = df['label'].astype(str)
    df_filtered = df[df['label'].notna() & (df['label'] != "") & (df['label'] != "nan")]

    if df_filtered.empty:
        st.error("No documents with valid labels found.")
        return

    distinct_labels = df_filtered['label'].unique().tolist()

    st.sidebar.title("Configurações")
    selected_label = st.sidebar.selectbox("Selecione um rótulo", distinct_labels)
    threshold = st.sidebar.slider("Threshold de Similaridade", 0.999, 1.0, 0.9999, step=0.0001)
    submit_button = st.sidebar.button("Submeter")

    if submit_button:
        st.subheader(f"Documentos Similares para o Rótulo: {selected_label}")

        # Filter documents for the selected label
        df_selected_label = df_filtered[df_filtered['label'] == selected_label].copy()

        if df_selected_label.empty:
            st.info(f"No documents found for the label: {selected_label}")
            return

        # Calculate pairwise cosine similarity
        embeddings = np.array(df_selected_label['embedding'].tolist())
        
        # Handle cases with single embedding to avoid errors in cosine_similarity
        if embeddings.shape[0] == 1:
            st.info("Only one document found for this label. Cannot calculate similarity.")
            return

        # Calculate cosine similarity matrix
        similarity_matrix = cosine_similarity(embeddings)

        # Find pairs with similarity above the threshold
        similar_pairs = []
        for i in range(len(similarity_matrix)):
            for j in range(i + 1, len(similarity_matrix)):
                if similarity_matrix[i, j] >= threshold:
                    similar_pairs.append({
                        'doc1_id': df_selected_label.iloc[i]['id_doc'],
                        'doc1_text': df_selected_label.iloc[i]['combined_text'],
                        'doc2_id': df_selected_label.iloc[j]['id_doc'],
                        'doc2_text': df_selected_label.iloc[j]['combined_text'],
                        'similarity': similarity_matrix[i, j]
                    })

        if similar_pairs:
            df_similar_pairs = pd.DataFrame(similar_pairs)
            df_similar_pairs = df_similar_pairs.sort_values(by='similarity', ascending=False)
            st.write(f"Found {len(df_similar_pairs)} pairs with similarity >= {threshold:.4f}:")
            
            for index, row in df_similar_pairs.iterrows():
                st.markdown(f"**Similarity: {row['similarity']:.4f}**")
                col1, col2 = st.columns(2)
                with col1:
                    st.text_area(f"Document 1 (ID: {row['doc1_id']})", row['doc1_text'], height=150, key=f"doc1_{index}")
                with col2:
                    st.text_area(f"Document 2 (ID: {row['doc2_id']})", row['doc2_text'], height=150, key=f"doc2_{index}")
                st.markdown("---")